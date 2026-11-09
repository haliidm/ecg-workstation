"""Launch the Windows bundle or run its local release verification."""
import sys
from pathlib import Path

from ecg_workstation import Workstation


def verify(report, record_path, checkpoint=None):
    import traceback
    from ecg_workstation import analyze, classify_record_beats, read_record

    app = None
    try:
        app = Workstation()
        app.update()
        app.record = read_record(record_path)
        app.results, app.axes = analyze(app.record)
        if not any('QRS' in result for result in app.results.values()):
            raise RuntimeError(f'No measured lead: {app.results}')
        app.beats, lead = classify_record_beats(app.record, app.beat_classifier)
        if not app.beats:
            raise RuntimeError('No beats detected')
        app.refresh()
        app.update()
        if len(app.fig.axes) != 13 or app.fig.axes[0].get_xlim() != (0, 10):
            raise RuntimeError('The 12+1 / 10 s plot did not render')
        checks = f'PASS: WFDB, morphology, {len(app.beats)} classified beats ({lead}), and 12+1 display'
        if checkpoint:
            app.founder.path = Path(checkpoint)
            app.founder.predict(app.record)
            checks += ', ECGFounder inference'
        Path(report).write_text(checks + '\n', encoding='utf-8')
        return 0
    except Exception:
        Path(report).write_text(traceback.format_exc(), encoding='utf-8')
        return 1
    finally:
        if app is not None:
            app.destroy()


if __name__ == '__main__':
    if sys.argv[1:2] == ['--verify-release']:
        sys.exit(verify(*sys.argv[2:]))
    Workstation().mainloop()
