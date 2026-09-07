"""Package the verified PDFs and editable manuscript sources, excluding patient data."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

base = Path(__file__).resolve().parent
out = base.parents[1] / 'output' / 'pdf'
out.mkdir(parents=True, exist_ok=True)
delivered = {}
for venue in ('soict', 'rivf'):
    source = base / venue / 'main.pdf'
    target = out / f'XBoneNet_{venue.upper()}_2026_Draft.pdf'
    shutil.copy2(source, target)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(source) == digest(target)
    delivered[venue] = {'file': str(target), 'bytes': target.stat().st_size,
                        'sha256': digest(target)}

archive = base / 'XBoneNet_2026_LaTeX.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
    for venue in ('soict', 'rivf'):
        for path in sorted((base / venue).iterdir()):
            if path.suffix in {'.tex', '.bib', '.bbl', '.cls', '.bst'}:
                z.write(path, path.relative_to(base).as_posix())
    for name in ('README_VI.md', 'figure_alt_text.md', 'build.ps1',
                 'audit_evidence.py', 'prepare_assets.py', 'qa_pdf.py',
                 'package_deliverables.py',
                 'evidence/classification_aggregate.csv',
                 'evidence/classification_per_seed.csv',
                 'evidence/statistics_verification.json',
                 'evidence/pdf_validation.json'):
        z.write(base / name, name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert all(not n.endswith(('.npz', '.pth', '.png')) for n in z.namelist())
    delivered['source_archive'] = {'file': str(archive), 'entries': len(z.namelist()),
                                   'bytes': archive.stat().st_size}
(base / 'evidence' / 'delivery_manifest.json').write_text(
    json.dumps(delivered, indent=2), encoding='utf-8')
print(json.dumps(delivered, indent=2))
