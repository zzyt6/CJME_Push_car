# 赛道图：663 × 189，8 pt 字体

- `composite_course_663x189_8pt.pdf`：论文排版用矢量图，字体已嵌入。
- `composite_course_663x189_8pt.png`：精确 663 × 189 像素，按 96 dpi 导出。
- `composite_course_663x189_8pt.svg`：可编辑矢量图，画布为 663 × 189 CSS 像素。

第二张参考图的尺寸按像素解释，采用 96 dpi 换算。最终物理尺寸为 **175.41875 × 50.00625 mm**，PDF 页面为 **497.25 × 141.75 pt**。所有英文、数字、坐标轴单位和标注均为 **8 pt**；这里的 8 pt 指磅值。

插入论文时保留上述原始尺寸（100%），可保持 8 pt 字号。缩放整张图会同时缩放文字。LaTeX 中可直接使用 `\includegraphics{composite_course_663x189_8pt.pdf}`，无需再指定缩放。

采用原图的几何记录 `docs/images/composite_course_geometry.json` 重绘，保留路线、方向箭头、颜色和英文说明，横纵坐标的米制比例一致。输出时不进行自动裁边，因此不会改变目标尺寸。

重新生成：

```powershell
& 'E:\envs\RLisaac601\python.exe' 'scripts\plot_composite_course.py'
```
