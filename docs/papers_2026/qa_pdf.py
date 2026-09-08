from pathlib import Path
import json, re
from PIL import Image, ImageOps, ImageDraw
from pypdf import PdfReader

BASE=Path(__file__).resolve().parent
report={}
for venue in ['soict','rivf']:
    doc=PdfReader(BASE/venue/'main.pdf')
    pages=[p.extract_text() for p in doc.pages]
    refs=[i+1 for i,t in enumerate(pages) if re.search(r'\b(?:References|REFERENCES)\b',t)]
    fonts=set()
    for p in doc.pages:
        for f in p['/Resources'].get_object().get('/Font',{}).get_object().values():
            fonts.add(str(f.get_object().get('/Subtype')))
    log=(BASE/venue/'main.log').read_text(encoding='utf-8',errors='replace')
    issues=[x for x in log.splitlines() if any(k in x for k in ['Overfull','undefined','multiply defined','LaTeX Error'])]
    report[venue]={'pages':len(pages),'references_begin_page':refs[0],
        'page_size_points':[float(x) for x in doc.pages[0].mediabox],
        'fonts':sorted(fonts),'log_issues':issues,
        'abstract_words':len(re.findall(r'\b[\w-]+\b',re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',(BASE/venue/'main.tex').read_text(encoding='utf-8'),re.S)[1]))}
    assert not issues, issues
    assert '/Type3' not in fonts
    assert len(pages)<=6 if venue=='rivf' else refs[0]-1<=12
    (BASE/'qa'/f'{venue}_extracted.txt').write_text('\n\n'.join(f'PAGE {i+1}\n{t}' for i,t in enumerate(pages)),encoding='utf-8')
    images=sorted((BASE/'qa').glob(f'{venue}-*.png'),key=lambda x:int(x.stem.split('-')[-1]))
    for start in range(0,len(images),4):
        sheet=Image.new('RGB',(1600,2280),'#dddddd')
        draw=ImageDraw.Draw(sheet)
        for off,path in enumerate(images[start:start+4]):
            im=Image.open(path).convert('RGB');im.thumbnail((780,1100))
            x=(off%2)*800+10;y=(off//2)*1140+30
            sheet.paste(im,(x,y))
            draw.text((x,y-22),path.stem,fill='black')
        sheet.save(BASE/'qa'/f'contact_{venue}_{start//4+1}.png')
(BASE/'evidence/pdf_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
