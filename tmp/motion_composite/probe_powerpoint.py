from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
ns={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
with ZipFile('tmp/motion_composite/final_candidate.pptx') as z:
    parts={n:z.read(n) for n in z.namelist()}
for i in range(1,7):
    name=f'ppt/charts/chart{i}.xml'
    node=E.fromstring(parts[name])
    m=node.find('.//c:plotArea/c:layout/c:manualLayout',ns)
    for mode in ['xMode','yMode']: m.find('c:'+mode,ns).set('val','factor')
    for tx in node.findall('.//c:txPr',ns):
        default=tx.find('.//a:defRPr',ns)
        end=tx.find('.//a:endParaRPr',ns)
        end.set('sz',default.get('sz'))
        E.SubElement(end,'{'+ns['a']+'}latin',typeface='Arial')
    parts[name]=E.tostring(node,xml_declaration=True,encoding='UTF-8',standalone=True)
with ZipFile('tmp/motion_composite/powerpoint_factor.pptx','w',ZIP_DEFLATED) as z:
    for n,b in parts.items(): z.writestr(n,b)
