from pathlib import Path
from xml.etree import ElementTree as ET
import json

import cv2
import numpy as np
from PIL import Image

from poster_vector_rebuilder.semantic_primitives import reconstruct_semantic_primitives


def _fixture(path: Path, mask_path: Path):
    img = np.full((300, 420, 3), 255, dtype=np.uint8)
    mask = np.zeros((300, 420), dtype=np.uint8)
    cv2.rectangle(img,(20,20),(120,80),(20,40,220),-1); cv2.rectangle(mask,(20,20),(120,80),255,-1)
    cv2.rectangle(img,(175,20),(255,85),(30,180,60),-1); cv2.rectangle(mask,(175,20),(255,85),255,-1)
    cv2.rectangle(img,(160,35),(270,70),(30,180,60),-1); cv2.rectangle(mask,(160,35),(270,70),255,-1)
    for c in [(175,35),(255,35),(175,70),(255,70)]:
        cv2.circle(img,c,15,(30,180,60),-1); cv2.circle(mask,c,15,255,-1)
    cv2.circle(img,(340,55),35,(200,40,40),-1); cv2.circle(mask,(340,55),35,255,-1)
    cv2.ellipse(img,(85,160),(55,30),20,0,360,(140,40,180),-1); cv2.ellipse(mask,(85,160),(55,30),20,0,360,255,-1)
    poly=np.array([[180,130],[260,165],[175,205]],np.int32)
    cv2.fillPoly(img,[poly],(200,130,20)); cv2.fillPoly(mask,[poly],255)
    cv2.line(img,(300,135),(395,190),(40,40,40),5); cv2.line(mask,(300,135),(395,190),255,5)
    cv2.circle(img,(330,250),35,(0,120,210),-1); cv2.circle(mask,(330,250),35,255,-1)
    cv2.circle(img,(330,250),16,(255,255,255),-1); cv2.circle(mask,(330,250),16,0,-1)
    pts=np.array([[35,235],[55,220],[80,225],[100,242],[95,270],[70,282],[45,270],[25,252]],np.int32)
    cv2.fillPoly(img,[pts],(20,170,170)); cv2.fillPoly(mask,[pts],255)
    Image.fromarray(img).save(path); Image.fromarray(mask).save(mask_path)


def test_semantic_tags_and_compound(tmp_path):
    src=tmp_path/'fixture.png'; mask=tmp_path/'mask.png'; out=tmp_path/'semantic.svg'
    _fixture(src,mask)
    report=reconstruct_semantic_primitives(src,out,mask_path=mask,colors=8,min_area=20,simplify=0.003)
    tags=[e.tag.split('}')[-1] for e in ET.parse(out).getroot().iter()]
    for tag in ('rect','circle','ellipse','polygon','line','path'):
        assert tag in tags
    counts=report['primitive_counts']
    for primitive in ('rectangle','rounded_rectangle','circle','ellipse','polygon','line','compound'):
        assert counts.get(primitive,0)>=1
    assert report['generic_path_ratio'] < 0.30


def test_report_is_serialized(tmp_path):
    src=tmp_path/'simple.png'; mask=tmp_path/'mask.png'; out=tmp_path/'out.svg'
    img=np.full((120,160,3),255,dtype=np.uint8); m=np.zeros((120,160),np.uint8)
    cv2.rectangle(img,(20,20),(140,100),(0,0,0),-1); cv2.rectangle(m,(20,20),(140,100),255,-1)
    Image.fromarray(img).save(src); Image.fromarray(m).save(mask)
    report=reconstruct_semantic_primitives(src,out,mask_path=mask,colors=2,min_area=10)
    saved=json.loads(Path(report['outputs']['report']).read_text())
    assert saved['schema']=='poster-vector-semantic-primitives-v1'
    assert saved['primitive_counts']['rectangle']>=1
    assert '<image' not in out.read_text().lower()
