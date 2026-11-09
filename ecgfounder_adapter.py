"""ECGFounder 150-class adapter; follows ptbxl_eval.py normalization/interpolation."""
import os
import json
from pathlib import Path
import importlib.util
import numpy as np

BUNDLED_ROOT = Path(__file__).resolve().parent / 'vendor' / 'ecgfounder'
DEFAULT_ROOT = BUNDLED_ROOT if BUNDLED_ROOT.is_dir() else Path('D:/ECG/ECGFounder-master/ECGFounder-master')
ROOT = Path(os.environ.get('ECGFOUNDER_ROOT', str(DEFAULT_ROOT)))

def load_checkpoint(path):
    """Permit only NumPy scalar metadata in the original training checkpoint."""
    import torch
    from numpy.core.multiarray import scalar
    allowed = [(scalar, 'numpy.core.multiarray.scalar'), np.dtype,
               type(np.dtype('float64')), type(np.dtype('float32'))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location='cpu', weights_only=True)

class Founder:
    def __init__(self):
        self.path = ROOT / 'checkpoint' / '12_lead_ECGFounder.pth'
        local = Path(__file__).parent / '12_lead_ECGFounder.pth'
        if local.is_file(): self.path = local
        self.model = None

    def predict(self, record):
        if not self.path.is_file():
            raise FileNotFoundError('未找到 ECGFounder 权重，请点击“模型权重”选择 12_lead_ECGFounder.pth')
        import torch
        from scipy.interpolate import interp1d
        names = [str(n).strip().upper() for n in record.sig_name]
        order = ['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
        missing = [n for n in order if names.count(n) != 1]
        if missing: raise ValueError('缺失或重复导联：' + ', '.join(missing))
        count = int(round(record.fs * 10))
        if record.sig_len < count: raise ValueError('ECGFounder 需要至少10秒完整12导联数据')
        data = record.p_signal[:count, [names.index(n) for n in order]].T.copy()
        for i,n in enumerate(order):
            unit = record.units[names.index(n)].lower()
            if unit in ('uv','µv'): data[i] /= 1000
            elif unit == 'v': data[i] *= 1000
            elif unit != 'mv': raise ValueError('不支持的单位：'+unit)
        if not np.isfinite(data).all(): raise ValueError('模型输入存在缺失值')
        if np.any(np.std(data, axis=1) < 1e-8): raise ValueError('存在平直导联，无法推理')
        data = (data-data.mean())/(data.std()+1e-8)
        data = interp1d(np.linspace(0,10,count),data,axis=1)(np.linspace(0,10,5000)).astype('float32')
        if self.model is None:
            spec = importlib.util.spec_from_file_location('ecgfounder_net', ROOT/'net1d.py')
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            self.labels = (ROOT/'tasks.txt').read_text(encoding='utf-8').splitlines()
            if len(self.labels)!=150: raise ValueError('标签文件必须包含150个类别')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = module.Net1D(in_channels=12,base_filters=64,ratio=1,filter_list=[64,160,160,400,400,1024,1024],m_blocks_list=[2,2,2,3,3,4,4],kernel_size=16,stride=2,groups_width=16,verbose=False,use_bn=False,use_do=False,n_classes=150)
            checkpoint = load_checkpoint(self.path)
            model.load_state_dict(checkpoint['state_dict'],strict=True)
            self.model = model.to(self.device).eval()
        with torch.inference_mode():
            scores = self.model(torch.from_numpy(data).unsqueeze(0).to(self.device)).sigmoid().cpu().numpy()[0]
        translations = json.loads((Path(__file__).parent/'founder_labels_zh.json').read_text(encoding='utf-8'))
        selected = [i for i in np.argsort(scores)[::-1] if scores[i] > 0.90]
        lines = [f'{translations[self.labels[i]]}：{scores[i]:.1%}' for i in selected]
        return 'ECGFounder · 前10秒\n模型概率 > 90%\n\n' + ('\n'.join(lines) if lines else '无概率高于90%的项目')
