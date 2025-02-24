# OptiMatchLocator

**项目名称：基于LightGlue的遥感图像匹配与地理坐标计算**

**项目描述**

/************

**依赖环境**

​	•	Python 3.8及以上

​	•	PyTorch 1.8.1及以上

​	•	OpenCV

​	•	GDAL

​	•	NumPy

​	•	PIL

​	•	LightGlue库（包含LightGlue和SuperPoint模型）

​	•	其他Python依赖库（详见requirements.txt）



**安装light glue**

```
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
```

**参数说明**

运行脚本时，可以通过命令行参数传入以下配置项：

​	•	--device：选择运行设备（默认为auto，支持cuda、cpu、mps等选项）

​	•	--num_keypoints：设定每张图像提取的关键点数量（默认为1024）

​	•	--image_ste_path：卫星底图影像路径

​	•	--image_uav_path：无人机影像目录路径（无人机图像命名规则为1.tif,，2.tif，3.tif……）

​	•	--save_path：输出结果保存路径

​	•	--fault_path：故障匹配图像保存路径

​	•	--start_lon：第一张无人机影像起始经度

​	•	--start_lat：第一张无人机影像起始纬度

​	•	--test_num：最大测试的图像数量（默认为10）

例如：

```
python main.py --device cuda --image_ste_path "path/to/satellite_image.tif" --image_uav_path "path/to/uav_images/" --save_path "path/to/save_results/" --fault_path "path/to/fault_path"
```

**项目结构**

```
├── main.py                # 主程序入口
├── requirements.txt       # 依赖库
├── utils/                 # 工具函数和模块
│   ├── pair_util.py       # 图像配对和匹配相关函数
│   ├── logger.py          # 日志记录功能
│   └── ...                # 其他工具函数
├── lightglue/             # LightGlue模型和相关实现
│   ├── LightGlue.py       # LightGlue实现
│   ├── SuperPoint.py      # SuperPoint特征提取模型实现
│   └── ...                # 其他LightGlue实现
└── output/                # 输出目录
```

**使用说明**

**1. 图像匹配**

​	•	本项目使用LightGlue和SuperPoint模型进行图像匹配，首先通过SuperPoint提取特征点，接着使用LightGlue进行特征匹配。

​	•	每次处理一对卫星图像和无人机图像，并通过图像配对计算出图像中的对应关系。



**2. 坐标转换与裁剪**

​	•	项目中提供了crop_geotiff_by_center_point函数，用于基于给定的地理坐标裁剪GeoTIFF图像。

​	•	通过geo2pixel函数，将地理坐标转换为图像的像素坐标，从而实现对特定区域的裁剪。



**3. 结果输出**

​	•	配对结果和坐标信息将保存在output目录下。

​	•	图像匹配结果将以可视化的形式保存在save_path和fault_path中。

​	•	每张图像的地理坐标将被记录到CSV文件中（image_coordinates.csv）。



**4. 错误处理**

​	•	若匹配失败，程序会通过边界拓展的方式自动调整搜索范围，尝试多次匹配，直到找到有效的匹配。



**5.示例输出**

​	•	日志文件log.txt会记录每一步的处理时间和性能。

​	•	CSV文件image_coordinates.csv会保存每张图像对应的地理坐标信息。



**6.函数说明**

​	•	geo2pixel(geotransform, lon, lat)：将经纬度转换为图像的像素坐标。

​	•	save_coordinates_to_csv(csv_file, image_name, coord)：将图像名称和地理坐标保存到CSV文件。

​	•	crop_geotiff_by_center_point(longitude, latitude, input_tif_path, crop_size_px, crop_size_py)：根据中心点经纬度裁剪GeoTIFF图像。

​	•	parse_opt()：解析命令行参数。

​	•	main()：主程序入口，进行图像处理、匹配和结果保存。
