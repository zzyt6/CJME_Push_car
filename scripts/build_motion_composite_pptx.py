"""Build editable OOXML charts and vector bands for the motion composite.

The bundled presentation JS runtime is unavailable on this Windows host.
Use standard OOXML, embedded literal-data workbooks, and Microsoft PowerPoint
for application rendering. No chart is embedded as a raster image.
"""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import zipfile
from lxml import etree as E

ROOT=Path(__file__).resolve().parents[1]
NS={"a":"http://schemas.openxmlformats.org/drawingml/2006/main",
    "p":"http://schemas.openxmlformats.org/presentationml/2006/main",
    "c":"http://schemas.openxmlformats.org/drawingml/2006/chart",
    "r":"http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "s":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL="http://schemas.openxmlformats.org/package/2006/relationships"
CT="http://schemas.openxmlformats.org/package/2006/content-types"
EMU=36000
# Leave room for tick labels around the native chart. Values are millimetres.
CHART_FRAME_MM=(56.,42.)
PLOT_INSET_MM=(10.,3.)


def el(tag,parent=None,**attrs):
    prefix,name=tag.split(":")
    node=E.Element(f"{{{NS[prefix]}}}{name}")
    for key,value in attrs.items():node.set(key,str(value))
    if parent is not None:parent.append(node)
    return node


def xml(root):return E.tostring(root,xml_declaration=True,encoding="UTF-8",standalone=True)


def relationships(items):
    root=E.Element(f"{{{REL}}}Relationships",nsmap={None:REL})
    for rid,kind,target in items:
        E.SubElement(root,f"{{{REL}}}Relationship",Id=rid,Type=NS["r"]+"/"+kind,Target=target)
    return xml(root)


def transform(parent,rect):
    x,y,w,h=rect
    node=el("a:xfrm",parent)
    el("a:off",node,x=round(x*EMU),y=round(y*EMU))
    el("a:ext",node,cx=round(w*EMU),cy=round(h*EMU))


def color(parent,value,alpha=None):
    fill=el("a:solidFill",parent);rgb=el("a:srgbClr",fill,val=value.lstrip("#"))
    if alpha is not None:el("a:alpha",rgb,val=round(alpha*1e5))


def no_line(parent):el("a:noFill",el("a:ln",parent))


def line(parent,value="#555555",width=.65,dashed=False):
    node=el("a:ln",parent,w=round(width*12700));color(node,value)
    if dashed:el("a:prstDash",node,val="dash")


def tx_pr(parent,size=8,rot=0):
    node=el("c:txPr",parent)
    el("a:bodyPr",node,rot=rot);el("a:lstStyle",node)
    paragraph=el("a:p",node);pp=el("a:pPr",paragraph)
    rp=el("a:defRPr",pp,sz=round(size*100),b=0,i=0)
    color(rp,"#000000");el("a:latin",rp,typeface="Arial")
    el("a:endParaRPr",paragraph,lang="en-US")


def group_tree(parent):
    tree=el("p:spTree",parent);nv=el("p:nvGrpSpPr",tree)
    el("p:cNvPr",nv,id=1,name="");el("p:cNvGrpSpPr",nv);el("p:nvPr",nv)
    x=el("a:xfrm",el("p:grpSpPr",tree))
    for name,attrs in (("off",dict(x=0,y=0)),("ext",dict(cx=0,cy=0)),("chOff",dict(x=0,y=0)),("chExt",dict(cx=0,cy=0))):
        el("a:"+name,x,**attrs)
    return tree


def text_shape(tree,id,text,rect,size=8,rotate=False,align="ctr"):
    shape=el("p:sp",tree);nv=el("p:nvSpPr",shape)
    el("p:cNvPr",nv,id=id,name=text);el("p:cNvSpPr",nv,txBox=1);el("p:nvPr",nv)
    props=el("p:spPr",shape);transform(props,rect)
    geometry=el("a:prstGeom",props,prst="rect");el("a:avLst",geometry)
    el("a:noFill",props);no_line(props)
    body=el("p:txBody",shape)
    attrs=dict(wrap="none",anchor="ctr",lIns=0,rIns=0,tIns=0,bIns=0)
    if rotate:attrs["vert"]="vert270"
    el("a:bodyPr",body,**attrs);el("a:lstStyle",body)
    paragraph=el("a:p",body);el("a:pPr",paragraph,algn=align)
    # Use native baseline/subscript runs for the scientific axis symbols.
    if text in ("vx (m/s)","vy (m/s)","wz (rad/s)"):
        pieces=[("ω" if text[0]=="w" else "v",False),(text[1],True),(text[2:],False)]
    else:pieces=[(text,False)]
    for value,subscript in pieces:
        run=el("a:r",paragraph)
        rp=el("a:rPr",run,lang="en-US",sz=round(size*100),b=0,i=0)
        if subscript:rp.set("baseline","-25000");rp.set("sz",str(round(size*72)))
        color(rp,"#000000");el("a:latin",rp,typeface="Arial")
        el("a:t",run).text=value
    return shape


def legend_shapes(tree,first_id,panel):
    """Editable scientific legend with fixed Arial text and no auto-wrapping."""
    x,y,_,_=panel["plot_rect_mm"]
    shape=el("p:sp",tree);nv=el("p:nvSpPr",shape)
    el("p:cNvPr",nv,id=first_id,name=f"{panel['engine']} legend background")
    el("p:cNvSpPr",nv);el("p:nvPr",nv)
    props=el("p:spPr",shape);transform(props,[x+.7,y+.7,18.5,9.1])
    el("a:avLst",el("a:prstGeom",props,prst="rect"))
    color(props,"#FFFFFF",.9);no_line(props)
    next_id=first_id+1
    for j,series in enumerate(panel["series"]):
        cy=y+2.2+j*2.7
        shape=el("p:sp",tree);nv=el("p:nvSpPr",shape)
        el("p:cNvPr",nv,id=next_id,name=f"{series['name']} legend line");next_id+=1
        el("p:cNvSpPr",nv);el("p:nvPr",nv)
        props=el("p:spPr",shape);transform(props,[x+1.4,cy,3.8,0])
        el("a:avLst",el("a:prstGeom",props,prst="line"))
        line(props,series["color"],.9,series["dashed"])
        text_shape(tree,next_id,series["name"],[x+6.1,cy-1.35,12,2.7],6.6,align="l")
        next_id+=1
    return next_id


def picture(tree,id,record,rid):
    shape=el("p:pic",tree);nv=el("p:nvPicPr",shape)
    el("p:cNvPr",nv,id=id,name=Path(record["path"]).name);el("p:cNvPicPr",nv);el("p:nvPr",nv)
    fill=el("p:blipFill",shape);blip=el("a:blip",fill)
    blip.set(f"{{{NS['r']}}}embed",rid)
    el("a:fillRect",el("a:stretch",fill))
    props=el("p:spPr",shape);transform(props,record["rect_mm"])
    el("a:avLst",el("a:prstGeom",props,prst="rect"))


def band_shape(tree,id,panel,band):
    rect=panel["plot_rect_mm"];_,_,w,h=rect
    x0,x1=panel["xlim"];y0,y1=panel["ylim"]
    points=[(round((x-x0)/(x1-x0)*w*EMU),round((y1-y)/(y1-y0)*h*EMU)) for x,y in band["vertices"]]
    shape=el("p:sp",tree);nv=el("p:nvSpPr",shape)
    el("p:cNvPr",nv,id=id,name=f"{panel['engine']} {panel['axis']} {band['name']} +/- SD")
    el("p:cNvSpPr",nv);el("p:nvPr",nv)
    props=el("p:spPr",shape);transform(props,rect)
    geom=el("a:custGeom",props)
    for name in ("avLst","gdLst","ahLst","cxnLst"):el("a:"+name,geom)
    el("a:rect",geom,l="0",t="0",r="r",b="b")
    path=el("a:path",el("a:pathLst",geom),w=round(w*EMU),h=round(h*EMU),stroke=0)
    for index,(x,y) in enumerate(points):
        node=el("a:moveTo" if index==0 else "a:lnTo",path);el("a:pt",node,x=x,y=y)
    el("a:close",path);color(props,band["color"],band["alpha"]);no_line(props)


def literal_workbook(series):
    # A new workbook snapshot of complete literal chart data, not formulas.
    workbook=el("s:workbook");sheets=el("s:sheets",workbook)
    sheet=el("s:sheet",sheets,name="Data",sheetId=1);sheet.set(f"{{{NS['r']}}}id","rId1")
    worksheet=el("s:worksheet");rows=el("s:sheetData",worksheet)
    n=max(len(s["x"]) for s in series)
    for i in range(n+1):
        row=el("s:row",rows,r=i+1)
        for j,s in enumerate(series):
            for k,kind in enumerate(("x","y")):
                col=chr(65+2*j+k);ref=f"{col}{i+1}"
                if i==0:
                    cell=el("s:c",row,r=ref,t="inlineStr")
                    el("s:t",el("s:is",cell)).text=("Time (s)" if k==0 else s["name"])
                elif i<=len(s[kind]):
                    cell=el("s:c",row,r=ref);el("s:v",cell).text=format(s[kind][i-1],".17g")
    types=E.Element(f"{{{CT}}}Types",nsmap={None:CT})
    for ext,ctype in (("rels","application/vnd.openxmlformats-package.relationships+xml"),("xml","application/xml")):
        E.SubElement(types,f"{{{CT}}}Default",Extension=ext,ContentType=ctype)
    for part,ctype in (("/xl/workbook.xml","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"),
                       ("/xl/worksheets/sheet1.xml","application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml")):
        E.SubElement(types,f"{{{CT}}}Override",PartName=part,ContentType=ctype)
    output=io.BytesIO()
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as z:
        for name,value in {"[Content_Types].xml":xml(types),"_rels/.rels":relationships([("rId1","officeDocument","xl/workbook.xml")]),
                           "xl/workbook.xml":xml(workbook),"xl/_rels/workbook.xml.rels":relationships([("rId1","worksheet","worksheets/sheet1.xml")]),
                           "xl/worksheets/sheet1.xml":xml(worksheet)}.items():z.writestr(name,value)
    return output.getvalue()


def numeric_reference(parent,tag,formula,values):
    ref=el("c:numRef",el("c:"+tag,parent));el("c:f",ref).text=formula
    cache=el("c:numCache",ref);el("c:formatCode",cache).text="General"
    el("c:ptCount",cache,val=len(values))
    for index,value in enumerate(values):el("c:v",el("c:pt",cache,idx=index)).text=format(value,".17g")


def manual_layout(parent,x,y,w,h):
    manual=el("c:manualLayout",el("c:layout",parent))
    el("c:layoutTarget",manual,val="inner")
    for key in ("xMode","yMode","wMode","hMode"):
        el("c:"+key,manual,val="edge" if key in ("xMode","yMode") else "factor")
    for key,val in zip(("x","y","w","h"),(x,y,w,h)):el("c:"+key,manual,val=val)


def chart_xml(panel,index):
    space=el("c:chartSpace");el("c:date1904",space,val=0);el("c:lang",space,val="en-US")
    chart=el("c:chart",space);el("c:autoTitleDeleted",chart,val=1)
    plot=el("c:plotArea",chart)
    frame_w,frame_h=CHART_FRAME_MM
    inset_x,inset_y=PLOT_INSET_MM
    _,_,plot_w,plot_h=panel["plot_rect_mm"]
    manual_layout(plot,inset_x/frame_w,inset_y/frame_h,plot_w/frame_w,plot_h/frame_h)
    scatter=el("c:scatterChart",plot);el("c:scatterStyle",scatter,val="line")
    el("c:varyColors",scatter,val=0)
    for j,s in enumerate(panel["series"]):
        ser=el("c:ser",scatter);el("c:idx",ser,val=j);el("c:order",ser,val=j)
        el("c:v",el("c:tx",ser)).text=s["name"]
        line(el("c:spPr",ser),s["color"],.9,s["dashed"])
        el("c:symbol",el("c:marker",ser),val="none")
        for k,tag in enumerate(("xVal","yVal")):
            col=chr(65+2*j+k);values=s[("x","y")[k]]
            numeric_reference(ser,tag,f"Data!${col}$2:${col}${len(values)+1}",values)
        el("c:smooth",ser,val=0)
    for aid in (index*10+1,index*10+2):el("c:axId",scatter,val=aid)
    for dim in (0,1):
        aid=index*10+dim+1;other=index*10+(2 if dim==0 else 1)
        axis=el("c:valAx",plot);el("c:axId",axis,val=aid)
        low,high=panel["xlim" if dim==0 else "ylim"]
        scale=el("c:scaling",axis);el("c:orientation",scale,val="minMax")
        el("c:max",scale,val=high);el("c:min",scale,val=low)
        el("c:delete",axis,val=0);el("c:axPos",axis,val="b" if dim==0 else "l")
        line(el("c:spPr",el("c:majorGridlines",axis)),"#DEDEDE",.4)
        el("c:numFmt",axis,formatCode="0" if dim==0 else "0.0",sourceLinked=0)
        el("c:majorTickMark",axis,val="out");el("c:minorTickMark",axis,val="none")
        el("c:tickLblPos",axis,val="nextTo");line(el("c:spPr",axis))
        tx_pr(axis,8);el("c:crossAx",axis,val=other)
        el("c:crossesAt",axis,val=panel["ylim" if dim==0 else "xlim"][0])
        el("c:crossBetween",axis,val="midCat")
        ticks=panel["xticks" if dim==0 else "yticks"]
        el("c:majorUnit",axis,val=round(ticks[1]-ticks[0],8))
    props=el("c:spPr",plot);el("a:noFill",props);no_line(props)
    el("c:plotVisOnly",chart,val=1);el("c:dispBlanksAs",chart,val="gap")
    props=el("c:spPr",space);el("a:noFill",props);no_line(props);tx_pr(space,8)
    external=el("c:externalData",space);external.set(f"{{{NS['r']}}}id","rId1")
    el("c:autoUpdate",external,val=0)
    return xml(space)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT/"logs/Picture/motion_composite")
    parser.add_argument("--output",type=Path,help="Write a separate revision without overwriting the previous deck")
    args=parser.parse_args();root=args.root.resolve()
    spec=json.loads((root/"data/composition.json").read_text(encoding="utf-8"))
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    parts={};overrides=[]
    presentation=el("p:presentation")
    master_id=el("p:sldMasterId",el("p:sldMasterIdLst",presentation),id=2147483648)
    master_id.set(f"{{{NS['r']}}}id","rId1")
    slide_id=el("p:sldId",el("p:sldIdLst",presentation),id=256)
    slide_id.set(f"{{{NS['r']}}}id","rId2")
    el("p:sldSz",presentation,cx=round(spec["size_mm"][0]*EMU),cy=round(spec["size_mm"][1]*EMU),type="custom")
    el("p:notesSz",presentation,cx=6858000,cy=9144000)
    parts["ppt/presentation.xml"]=xml(presentation)
    parts["_rels/.rels"]=relationships([("rId1","officeDocument","ppt/presentation.xml")])
    parts["ppt/_rels/presentation.xml.rels"]=relationships([
        ("rId1","slideMaster","slideMasters/slideMaster1.xml"),("rId2","slide","slides/slide1.xml")])
    master=el("p:sldMaster");group_tree(el("p:cSld",master))
    el("p:clrMap",master,bg1="lt1",tx1="dk1",bg2="lt2",tx2="dk2",
       accent1="accent1",accent2="accent2",accent3="accent3",accent4="accent4",accent5="accent5",accent6="accent6",hlink="hlink",folHlink="folHlink")
    layout_id=el("p:sldLayoutId",el("p:sldLayoutIdLst",master),id=2147483649)
    layout_id.set(f"{{{NS['r']}}}id","rId1")
    styles=el("p:txStyles",master)
    for name in ("titleStyle","bodyStyle","otherStyle"):
        rp=el("a:defRPr",el("a:lvl1pPr",el("p:"+name,styles)),sz=800,b=0,i=0)
        el("a:latin",rp,typeface="Arial")
    parts["ppt/slideMasters/slideMaster1.xml"]=xml(master)
    parts["ppt/slideMasters/_rels/slideMaster1.xml.rels"]=relationships([
        ("rId1","slideLayout","../slideLayouts/slideLayout1.xml"),("rId2","theme","../theme/theme1.xml")])
    layout=el("p:sldLayout",type="blank",preserve=1);group_tree(el("p:cSld",layout,name="Blank"))
    el("a:masterClrMapping",el("p:clrMapOvr",layout))
    parts["ppt/slideLayouts/slideLayout1.xml"]=xml(layout)
    parts["ppt/slideLayouts/_rels/slideLayout1.xml.rels"]=relationships([
        ("rId1","slideMaster","../slideMasters/slideMaster1.xml")])
    with zipfile.ZipFile(manifest["reference_pptx"]) as source:
        theme=E.fromstring(source.read("ppt/theme/theme1.xml"))
    for font in theme.findall(".//a:fontScheme//a:latin",NS):font.set("typeface","Arial")
    parts["ppt/theme/theme1.xml"]=xml(theme)
    slide=el("p:sld");tree=group_tree(el("p:cSld",slide,name="Motion ability: Isaac Sim and MuJoCo"))
    slide_rels=[("rId1","slideLayout","../slideLayouts/slideLayout1.xml")]
    shape_id=2
    for index,item in enumerate(spec["images"]):
        rid=f"rId{index+2}";media=f"motion{index+1}.png"
        parts["ppt/media/"+media]=Path(item["path"]).read_bytes()
        picture(tree,shape_id,item,rid);shape_id+=1
        slide_rels.append((rid,"image","../media/"+media))
    for text in spec["texts"]:
        width=50 if "motion" in text["text"] else 22
        text_shape(tree,shape_id,text["text"],[text["x_mm"]-width/2,text["y_mm"]-2.5,width,5],text["font_pt"])
        shape_id+=1
    for index,panel in enumerate(spec["panels"],1):
        for band in panel["bands"]:
            band_shape(tree,shape_id,panel,band);shape_id+=1
        x,y,w,h=panel["plot_rect_mm"]
        ylabel={"vx":"vx (m/s)","vy":"vy (m/s)","wz":"wz (rad/s)"}[panel["axis"]]
        text_shape(tree,shape_id,ylabel,[x-10.,y,4,h],8,True);shape_id+=1
        frame=el("p:graphicFrame",tree)
        nv=el("p:nvGraphicFramePr",frame)
        el("p:cNvPr",nv,id=shape_id,name=f"{panel['engine']} {panel['axis']} native data chart");shape_id+=1
        el("p:cNvGraphicFramePr",nv);el("p:nvPr",nv)
        trans=el("p:xfrm",frame)
        inset_x,inset_y=PLOT_INSET_MM
        el("a:off",trans,x=round((x-inset_x)*EMU),y=round((y-inset_y)*EMU))
        el("a:ext",trans,cx=round(CHART_FRAME_MM[0]*EMU),cy=round(CHART_FRAME_MM[1]*EMU))
        graph=el("a:graphic",frame);gd=el("a:graphicData",graph,uri=NS["c"])
        chart=el("c:chart",gd);rid=f"rId{100+index}";chart.set(f"{{{NS['r']}}}id",rid)
        slide_rels.append((rid,"chart",f"../charts/chart{index}.xml"))
        parts[f"ppt/charts/chart{index}.xml"]=chart_xml(panel,index)
        parts[f"ppt/charts/_rels/chart{index}.xml.rels"]=relationships([("rId1","package",f"../embeddings/chartData{index}.xlsx")])
        parts[f"ppt/embeddings/chartData{index}.xlsx"]=literal_workbook(panel["series"])
        overrides.append((f"/ppt/charts/chart{index}.xml","application/vnd.openxmlformats-officedocument.drawingml.chart+xml"))
        if panel["show_legend"]:
            shape_id=legend_shapes(tree,shape_id,panel)
    el("a:masterClrMapping",el("p:clrMapOvr",slide))
    parts["ppt/slides/slide1.xml"]=xml(slide)
    parts["ppt/slides/_rels/slide1.xml.rels"]=relationships(slide_rels)
    types=E.Element(f"{{{CT}}}Types",nsmap={None:CT})
    for ext,ctype in (("rels","application/vnd.openxmlformats-package.relationships+xml"),("xml","application/xml"),
                      ("png","image/png"),("xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")):
        E.SubElement(types,f"{{{CT}}}Default",Extension=ext,ContentType=ctype)
    for part,kind in (("presentation","presentation.main"),("slides/slide1","slide"),
                      ("slideMasters/slideMaster1","slideMaster"),("slideLayouts/slideLayout1","slideLayout")):
        overrides.append((f"/ppt/{part}.xml",f"application/vnd.openxmlformats-officedocument.presentationml.{kind}+xml"))
    overrides.append(("/ppt/theme/theme1.xml","application/vnd.openxmlformats-officedocument.theme+xml"))
    for name,ctype in overrides:E.SubElement(types,f"{{{CT}}}Override",PartName=name,ContentType=ctype)
    parts["[Content_Types].xml"]=xml(types)
    target=args.output.resolve() if args.output else root/"motion_ability_composite.pptx"
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as archive:
        for name,value in parts.items():archive.writestr(name,value)
    print(json.dumps(dict(pptx=str(target),native_charts=len(spec["panels"]),embedded_workbooks=len(spec["panels"]),
                          editable_bands=12,source_images=len(spec["images"])),indent=2))


if __name__=="__main__":main()
