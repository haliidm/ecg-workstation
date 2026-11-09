"""启动：python ecg_workstation.py。布局名称采用行×列。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / '.runtime_deps'))
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import concurrent.futures
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

LEADS = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
DISPLAY_SECONDS = 10
BEAT_COLORS = {'N': '#14803b', 'S': '#a87900', 'V': '#8129a3',
               'F': '#c51f2d', 'Q': '#616161', 'UNKNOWN': '#616161'}

def canonical_lead(name):
    """Match standard lead names regardless of case or surrounding whitespace."""
    cleaned = str(name).strip()
    return next((lead for lead in LEADS if lead.casefold() == cleaned.casefold()), cleaned)

def read_record(path):
    import wfdb
    rec = wfdb.rdrecord(str(Path(path).with_suffix('')))
    if rec.p_signal is None or not len(rec.p_signal):
        raise ValueError('记录没有有效波形数据')
    return rec


def signal_in_mv(record, channel):
    signal = np.array(record.p_signal[:, channel], dtype=float, copy=True)
    unit = record.units[channel].strip().casefold()
    factors = {'mv': 1, 'uv': 0.001, 'μv': 0.001, 'v': 1000}
    if unit not in factors:
        raise ValueError('不支持的幅值单位：' + record.units[channel])
    return signal * factors[unit]


def classify_record_beats(record, classifier):
    names = [canonical_lead(name) for name in record.sig_name]
    for lead in ('II', 'MLII', 'I', *LEADS[2:]):
        channels = [i for i, name in enumerate(names) if name.casefold() == lead.casefold()]
        if len(channels) != 1:
            continue
        try:
            signal = signal_in_mv(record, channels[0])
        except ValueError:
            continue
        if np.isfinite(signal).all() and np.ptp(signal) > 0:
            return classifier.predict(signal, record.fs), lead
    raise ValueError('没有可用于心搏分类的有效心电导联')

def analyze(record):
    np.float = float
    np.int = int
    from tools.QRSDetector import simple_qrs_detector
    from tools.SingleBeatBounds import dwt_ecg_delineator
    from features.ECGAvgBeat import extract_avg_wave, extract_features
    results = {}
    signals = {}
    references = {}
    for i, name in enumerate(record.sig_name):
        name = canonical_lead(name)
        try:
            signal = signal_in_mv(record, i)
            if not np.all(np.isfinite(signal)): raise ValueError('存在缺失采样')
            if name in LEADS[:6] and np.ptp(signal) > 0:
                signals[name] = signal
            peaks = simple_qrs_detector(signal, record.fs)
            if len(peaks) < 3: raise ValueError('有效心搏少于3个')
            wave, r, rr = extract_avg_wave(signal, peaks, record.fs)
            b = dwt_ecg_delineator(wave, r, record.fs)
            if name in LEADS[:6]:
                references[name] = (peaks, wave, r, b)
            f = extract_features(wave, b['ECG_P_Onset'], b['ECG_P_Offset'], b['ECG_R_Onset'], b['ECG_R_Offset'], b['ECG_T_Onset'], b['ECG_T_Offset'], record.fs, rr / record.fs * 1000)
            qt = (b['ECG_T_Offset'] - b['ECG_R_Onset']) / record.fs * 1000
            f['QT'], f['QTc'] = qt, qt / np.sqrt(rr / record.fs)
            f['T']['duration'] = (b['ECG_T_Offset'] - b['ECG_T_Onset']) / record.fs * 1000
            results[name] = f
        except Exception as e: results[name] = {'错误': str(e)}
    return results, calculate_record_axes(signals, references, record.fs)


def calculate_record_axes(signals, references, fs):
    from features.ECGAvgBeat import extract_avg_wave
    from features.QRSAxis import frontal_axes

    if 'I' not in signals:
        return {}
    inferior = next((name for name in ('aVF', 'II', 'III') if name in signals), None)
    if inferior is None:
        return {}

    axes = {}
    for reference in ('II', 'I', 'aVF', 'III', 'aVL', 'aVR'):
        if reference not in references:
            continue
        peaks, wave, r, bounds = references[reference]
        # Skip reference templates where an included beat used zero padding.
        if peaks[1] < r or peaks[-2] + len(wave) - r > len(signals['I']):
            continue
        lead_i = wave if reference == 'I' else extract_avg_wave(signals['I'], peaks, fs)[0]
        second = wave if reference == inferior else extract_avg_wave(signals[inferior], peaks, fs)[0]
        if inferior == 'II':
            second = second - lead_i / 2
        elif inferior == 'III':
            second = second + lead_i / 2
        for name, angle in frontal_axes(lead_i, second, bounds, r, fs).items():
            axes.setdefault(name, angle)
        if len(axes) == 3:
            break
    return axes

class Workstation(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('ECG Workstation · 心电分析工作站')
        self.geometry('1500x950')
        self.record = None
        self.results = {}
        self.axes = {}
        self.beats = []
        from beat_classifier import BeatClassifier
        self.beat_classifier = BeatClassifier()
        from ecgfounder_adapter import Founder
        self.founder = Founder()
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.busy = False
        top = ttk.Frame(self, padding=12); top.pack(fill='x')
        for title, command in [('新患者',self.import_record),('重新分析',self.start_analysis),('常规心电',self.draw)]:
            ttk.Button(top,text=title,command=command).pack(side='left',padx=5)
        self.patient = tk.StringVar(value='尚未导入患者记录')
        ttk.Button(top,text='模型权重',command=self.select_weights).pack(side='left',padx=5)
        ttk.Label(top,textvariable=self.patient).pack(side='left',padx=25)
        controls = ttk.Frame(self,padding=10); controls.pack(fill='x')
        self.layout = tk.StringVar(value='12+1'); self.speed = tk.StringVar(value='25'); self.gain = tk.StringVar(value='10'); self.lead = tk.StringVar(value='II')
        for label,var,values in [('布局（行×列）',self.layout,['4X3','3X4','6X2','12+1']),('走速 mm/s',self.speed,['12.5','25','50']),('增益 mm/mV',self.gain,['5','10','20']),('参数导联',self.lead,LEADS)]:
            ttk.Label(controls,text=label).pack(side='left',padx=6)
            box = ttk.Combobox(controls,textvariable=var,values=values,width=10,state='readonly'); box.pack(side='left'); box.bind('<<ComboboxSelected>>',lambda e:self.refresh())
        ttk.Label(controls,text='起始时间（秒）').pack(side='left',padx=8)
        self.start = tk.DoubleVar(value=0)
        ttk.Spinbox(controls,from_=0,to=86400,increment=1,textvariable=self.start,width=8,command=self.draw).pack(side='left')
        ttk.Button(controls,text='刷新',command=self.draw).pack(side='left',padx=5)
        self.beat_status = tk.StringVar(value='心搏分类 · 等待分析')
        beat_bar = ttk.Frame(self, padding=(12, 0, 12, 5))
        beat_bar.pack(fill='x')
        ttk.Label(beat_bar, textvariable=self.beat_status).pack(side='left', padx=(0, 14))
        for label, name in [('N', '正常'), ('S', '室上性'), ('V', '室性'),
                            ('F', '融合'), ('Q', 'Q类'), ('UNKNOWN', '未判定')]:
            symbol = '·' if label == 'UNKNOWN' else label
            ttk.Label(beat_bar, text=f'{symbol} {name}', foreground=BEAT_COLORS[label]).pack(side='left', padx=5)
        body = ttk.Panedwindow(self,orient='horizontal'); body.pack(fill='both',expand=True)
        side = ttk.Frame(body,padding=12); body.add(side,weight=0)
        ttk.Label(side,text='算法测量').pack(anchor='w')
        self.table = ttk.Treeview(side,columns=('value',),show='tree headings',height=24)
        self.table.heading('#0',text='参数'); self.table.heading('value',text='结果'); self.table.column('#0',width=165); self.table.column('value',width=100); self.table.pack(fill='both',expand=True)
        ttk.Label(side,text='神经网络诊断',font=('',12,'bold')).pack(anchor='w',pady=(20,8))
        self.diagnosis = tk.StringVar(value='ECGFounder · 等待加载权重和记录')
        ttk.Label(side,textvariable=self.diagnosis,wraplength=265).pack(anchor='w')
        ttk.Label(side,text='时限按所选导联显示。\n电轴为当前记录的额面估计值。',wraplength=260).pack(anchor='w',pady=20)
        plot = ttk.Frame(body); body.add(plot,weight=1)
        self.paper_view = tk.Canvas(plot, highlightthickness=0, background='white', yscrollincrement=24)
        scrollbar = ttk.Scrollbar(plot, orient='vertical', command=self.paper_view.yview)
        scrollbar.pack(side='right', fill='y')
        self.paper_view.configure(yscrollcommand=scrollbar.set)
        self.paper_view.pack(side='left', fill='both', expand=True)
        self.fig = Figure(figsize=(4, 4), dpi=100, facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.paper_view)
        paper_widget = self.canvas.get_tk_widget()
        self.paper_window = self.paper_view.create_window(0, 0, anchor='nw', window=paper_widget)
        self.resize_job = None
        self.paper_view.bind('<Configure>', self.queue_paper_resize)
        for widget in (paper_widget, self.paper_view):
            widget.bind('<MouseWheel>', self.scroll_paper)
        self.status = tk.StringVar(value='就绪 · 选择 .hea 文件；关联 .dat 文件需保留在同一目录')
        ttk.Label(self,textvariable=self.status,padding=8).pack(fill='x')
        self.draw()
    def run_job(self,fn,callback):
        if self.busy: return
        self.busy=True
        future=self.pool.submit(fn)
        def poll():
            if not future.done(): self.after(100,poll); return
            self.busy=False
            try: callback(future.result())
            except Exception as e:
                self.status.set('操作失败'); messagebox.showerror('无法完成',str(e))
        self.after(100,poll)
    def import_record(self):
        if self.busy: return
        path=filedialog.askopenfilename(title='新患者 · 导入心电记录',filetypes=[('WFDB 头文件','*.hea')])
        if not path:return
        self.status.set('正在读取心电数据…')
        def done(rec):
            self.record=rec; self.results={}; self.axes={}; self.beats=[]; self.start.set(0)
            self.paper_view.yview_moveto(0)
            self.patient.set(f'{rec.record_name}  |  {rec.n_sig} 导联  |  {rec.fs:g} Hz  |  {rec.sig_len/rec.fs:.1f} 秒')
            self.refresh(); self.start_analysis()
        self.run_job(lambda:read_record(path),done)
    def start_analysis(self):
        if self.record is None: messagebox.showinfo('提示','请先通过“新患者”导入 .hea 文件'); return
        if self.busy:return
        self.status.set('正在逐导联分析…')
        self.diagnosis.set('ECGFounder · 等待推理…')
        self.beat_status.set('心搏分类 · 正在分析')
        def done(result):
            (self.results, self.axes), diagnosis, self.beats, beat_status = result
            self.diagnosis.set(diagnosis)
            self.beat_status.set(beat_status)
            self.refresh(); self.status.set('处理完成 · 模型状态见左侧诊断区')
        def work():
            measurements = analyze(self.record)
            try:
                beats, lead = classify_record_beats(self.record, self.beat_classifier)
                beat_status = f'心搏分类 · {lead} 导联 · {len(beats)} 搏' if beats else '心搏分类 · 未检出心搏'
            except Exception as e:
                beats, beat_status = [], f'心搏分类未完成：{e}'
            try: diagnosis = self.founder.predict(self.record)
            except Exception as e: diagnosis = 'ECGFounder 未完成推理：\n' + str(e)
            return measurements, diagnosis, beats, beat_status
        self.run_job(work,done)
    def select_weights(self):
        if self.busy: return
        path = filedialog.askopenfilename(title='选择 ECGFounder 12导联150类权重',filetypes=[('PyTorch 权重','*.pth *.pt')])
        if path:
            self.founder.path = Path(path)
            self.founder.model = None
            self.diagnosis.set('权重已选择；重新分析时加载验证')
            if self.record is not None: self.start_analysis()
    def refresh(self):
        self.table.delete(*self.table.get_children())
        data = self.results.get(self.lead.get(), {})
        def value(group, key): return data.get(group, {}).get(key)
        def fmt(v):
            return f'{v:.2f}' if isinstance(v,(int,float)) and np.isfinite(v) else '未计算'
        rv5 = self.results.get('V5',{}).get('QRS',{}).get('RAmplitude')
        sv1 = self.results.get('V1',{}).get('QRS',{}).get('SAmplitude')
        total = rv5+sv1 if all(isinstance(v,(int,float)) and np.isfinite(v) for v in [rv5,sv1]) else None
        rows = [('心率 (bpm)',data.get('HR')),('P时限 (ms)',value('P','duration')),
                ('QRS时限 (ms)',value('QRS','QRSDuration')),('T时限 (ms)',value('T','duration')),
                ('PR间期 (ms)',value('PR','interval')),('QT间期 (ms)',data.get('QT')),
                ('QTc间期 (ms)',data.get('QTc')),('P轴 (°)',self.axes.get('P')),
                ('QRS轴 (°)',self.axes.get('QRS')),('T轴 (°)',self.axes.get('T'))]
        for name,v in rows: self.table.insert('', 'end',text=name,values=(fmt(v),))
        self.table.insert('', 'end',text='R(V5)/S(V1) (mV)',values=(fmt(rv5)+' / '+fmt(sv1),))
        self.table.insert('', 'end',text='R(V5)+S(V1) (mV)',values=(fmt(total),))
        if '错误' in data: self.status.set('当前导联分析失败：'+data['错误'])
        self.draw()

    def queue_paper_resize(self, _event):
        if self.resize_job is not None:
            self.after_cancel(self.resize_job)
        self.resize_job = self.after(80, self.finish_paper_resize)

    def finish_paper_resize(self):
        self.resize_job = None
        self.draw()

    def scroll_paper(self, event):
        self.paper_view.yview_scroll(-int(event.delta / 120) * 3, 'units')
        return 'break'

    def size_paper(self, width, height):
        self.paper_view.itemconfigure(self.paper_window, width=width, height=height)
        self.paper_view.configure(scrollregion=(0, 0, width, height))
        self.fig.set_size_inches(width / self.fig.dpi, height / self.fig.dpi, forward=False)

    def draw(self):
        self.fig.clear()
        width = self.paper_view.winfo_width()
        viewport_height = self.paper_view.winfo_height()
        if width < 120 or viewport_height < 60:
            return
        if self.record is None:
            self.size_paper(width, viewport_height)
            ax=self.fig.add_subplot(); ax.axis('off'); ax.text(.5,.55,'ECG WORKSTATION',ha='center',fontsize=26,color='#235e85'); ax.text(.5,.43,'Import a WFDB .hea record to begin',ha='center',fontsize=13); self.canvas.draw_idle(); return
        rows,cols=(13,1) if self.layout.get()=='12+1' else tuple(map(int,self.layout.get().split('X')))
        try:start=max(0,min(float(self.start.get()),max(0,self.record.sig_len/self.record.fs-DISPLAY_SECONDS)))
        except Exception:return
        speed,gain=float(self.speed.get()),float(self.gain.get())
        # Fit a fixed time span in each column and use its scale vertically too.
        top_margin, bottom_margin, side_margin = 28, 25, 10
        duration = DISPLAY_SECONDS
        panel_mm = int(duration * speed)
        panel_px = (width - 2 * side_margin) / cols
        pixels_per_mm = panel_px / panel_mm
        row_mm = max(30, 10 * int((viewport_height - top_margin - bottom_margin) / (rows * pixels_per_mm * 10)))
        row_px = row_mm * pixels_per_mm
        paper_height = max(viewport_height, int(np.ceil(rows * row_px + top_margin + bottom_margin)))
        paper_left = (width - cols * panel_px) / 2
        self.size_paper(width, paper_height)
        amplitude_limit = row_mm / (2 * gain)
        names=LEADS+(['II'] if rows==13 else [])
        record_leads = [canonical_lead(name) for name in self.record.sig_name]
        a = int(start * self.record.fs)
        z = min(self.record.sig_len, a + int(np.ceil(duration * self.record.fs)))
        visible_beats = [(peak, label) for peak, label in self.beats if a <= peak < z]
        for index,name in enumerate(names):
            row, col = divmod(index, cols)
            ax = self.fig.add_axes(((paper_left + col * panel_px) / width,
                                   (paper_height - top_margin - (row + 1) * row_px) / paper_height,
                                   panel_px / width, row_px / paper_height), facecolor='#fffafa')
            ax.set_xticks(np.arange(0, panel_mm + 1, 5) / speed)
            ax.set_xticks(np.arange(panel_mm + 1) / speed, minor=True)
            ax.set_yticks(np.arange(-row_mm // 2, row_mm // 2 + 1, 5) / gain)
            ax.set_yticks(np.arange(-row_mm // 2, row_mm // 2 + 1) / gain, minor=True)
            ax.set_xlim(0, duration)
            ax.set_ylim(-amplitude_limit, amplitude_limit)
            ax.grid(which='major',color='#e7a2aa',linewidth=.6); ax.grid(which='minor',color='#f5dce0',linewidth=.35)
            ax.tick_params(which='both',length=0,labelbottom=False,labelleft=False)
            ax.text(.012,.94,'II (Rhythm)' if index == 12 else name,transform=ax.transAxes,
                    va='top',fontsize=9,color='#263c48',fontweight='bold',
                    bbox={'facecolor':'#fffafa','edgecolor':'none','pad':1.5})
            if name in record_leads:
                channel=record_leads.index(name)
                y=self.record.p_signal[a:z,channel].copy(); unit=self.record.units[channel].lower()
                if unit in ('uv','µv'):y/=1000
                elif unit=='v':y*=1000
                if unit in ('mv','uv','µv','v'):ax.plot(np.arange(len(y))/self.record.fs,y,color='#263c48',linewidth=.65)
                else:ax.text(.4,.5,'Unsupported units',transform=ax.transAxes)
            else:ax.text(.4,.5,'Lead unavailable',transform=ax.transAxes,color='gray')
            if index < cols:
                for peak, label in visible_beats:
                    symbol = '·' if label == 'UNKNOWN' else label
                    ax.text((peak - a) / self.record.fs, 1.02, symbol,
                            transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                            fontsize=9, fontweight='bold', color=BEAT_COLORS[label])
            for spine in ax.spines.values():spine.set_visible(False)
        footer_y = (paper_height - top_margin - rows * row_px - 8) / paper_height
        self.fig.text(paper_left / width, footer_y, f'{speed:g} mm/s   |   {gain:g} mm/mV',
                      va='top', fontsize=9, color='#56656e')
        self.canvas.draw_idle()

if __name__=='__main__': Workstation().mainloop()
