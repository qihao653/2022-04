# 2022-04
# 基于像元的随机森林方法提取浮筏养殖区
pip install rasterio numpy scikit-learn matplotlib pandas geopandas tqdm scipy opencv-python scikit-image
# ===================== 江苏近海浮筏养殖区 基于像元的随机森林分类代码 =====================
# 开发环境：Python3.8+ ，适配ENVI、SARscape预处理后的Sentinel-1影像
# 核心功能：SAR影像预处理、12维特征集构建、基于像元的随机森林训练、浮筏提取、精度评价、时序批量处理
# ==========================================================================================
import rasterio
import numpy as np
import pandas as pd
import cv2
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import geopandas as gpd
from rasterio.features import geometry_mask
from scipy.ndimage import median_filter
# ===================== 1. 核心参数配置（与大创项目完全匹配）=====================
# 1.1 分类体系与编码
CLASS_DICT = {
    0: "海水背景",
    1: "浮筏养殖-紫菜筏式养殖",
    2: "浮筏养殖-贝类筏式养殖",
    3: "浮筏养殖-网箱养殖",
    4: "岸滩/陆地非养殖区"
}
CLASS_NUM = len(CLASS_DICT) - 1  
# 1.2 随机森林模型核心参数
RF_PARAMS = {
    "n_estimators": 150,  
    "criterion": "gini",  
    "max_depth": 12,  
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,  
    "n_jobs": -1,  
    "oob_score": True, 
    "verbose": 1
}
# 1.3 SAR影像波段配置（Sentinel-1 双极化数据，预处理后波段顺序：VV、VH）
BAND_NAMES = ["VV", "VH"]
BAND_INDEX = {name: i for i, name in enumerate(BAND_NAMES)}
# 1.4 文件路径配置
# 预处理后的SAR影像路径（单期/多期tif，需为辐射定标、地形校正、多视处理后的后向散射系数影像）
IMAGE_PATH = r"./sentinel1_data/202206_S1A_IW_GRD.tif"
# 样本ROI路径（shp格式，ENVI/ArcGIS绘制，需包含class字段对应上述分类编码）
SAMPLE_SHP_PATH = r"./roi/raft_aquaculture_sample.shp"
# 分类结果输出路径
OUTPUT_PATH = r"./classification_result"
# 精度报告输出路径
ACCURACY_PATH = r"./accuracy_report"
# 创建输出文件夹
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(ACCURACY_PATH, exist_ok=True)
# ===================== 2. SAR影像预处理（相干斑噪声抑制，解决项目中噪声问题）=====================
def sar_preprocessing(img_array, filter_size=3):
    
    print("开始SAR影像相干斑噪声抑制...")
    denoised_array = np.zeros_like(img_array, dtype=np.float32)
    
    
    for i in range(img_array.shape[0]):
        band_data = img_array[i, :, :]
      
        mean = cv2.blur(band_data, (filter_size, filter_size))
       
        var = cv2.blur(band_data ** 2, (filter_size, filter_size)) - mean ** 2
        
        noise_var = np.mean(var)
        
        weight = var / (var + noise_var + 1e-8)
        denoised_band = mean + weight * (band_data - mean)
        denoised_array[i, :, :] = denoised_band
    
    
    denoised_array = np.nan_to_num(denoised_array, nan=0, posinf=0, neginf=0)
    print("SAR影像预处理完成")
    return denoised_array
# ===================== 3. 特征集构建（复现项目中12维特征，极化+水体指数+纹理+极化分解）=====================
def build_feature_set(img_array):
    
    print("开始构建12维分类特征集...")
   
    VV = img_array[BAND_INDEX["VV"], :, :].astype(np.float32)
    VH = img_array[BAND_INDEX["VH"], :, :].astype(np.float32)
    epsilon = 1e-8  # 避免分母为0
    
    VV_VH_ratio = VV / (VH + epsilon)
   
    VH_VV_ratio = VH / (VV + epsilon)
    
    VV_VH_diff = VV - VH
   
    RVI = (4 * VH) / (VV + VH + epsilon)
    
    SWI = (VH - VV) / (VH + VV + epsilon)
    
    vh_normalized = cv2.normalize(VH, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    glcm = graycomatrix(vh_normalized, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
   
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    
    entropy = -np.sum(glcm * np.log2(glcm + epsilon))
   
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    asm = graycoprops(glcm, 'ASM')[0, 0]
    
    scatter_entropy = - (VV * np.log2(VV + epsilon) + VH * np.log2(VH + epsilon)) / 2
   
    feature_list = [
        VV, VH, VV_VH_ratio, VH_VV_ratio, VV_VH_diff,
        RVI, SWI, contrast, entropy, correlation, asm, scatter_entropy
    ]
    feature_array = np.stack(feature_list, axis=0)
    feature_names = [
        "VV", "VH", "VV/VH", "VH/VV", "VV-VH",
        "RVI", "SWI", "Contrast", "Entropy", "Correlation", "ASM", "Scatter_Entropy"
    ]
   
    feature_array = np.nan_to_num(feature_array, nan=0, posinf=0, neginf=0)
    print(f"特征集构建完成，共{len(feature_names)}维特征")
    return feature_array, feature_names
# ===================== 4. 基于像元的样本特征提取 =====================
def extract_pixel_samples(feature_array, sample_shp, transform):
   
    print("开始逐像元提取训练样本特征...")
    
    gdf = gpd.read_file(sample_shp)
    
    if "class" not in gdf.columns:
        raise ValueError("样本shp文件必须包含'class'字段，对应分类编码！")
   
    X = []
    y = []
   
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="提取像元样本"):
        class_label = row["class"]
        geometry = row["geometry"]
        
        mask = geometry_mask(
            [geometry],
            out_shape=feature_array.shape[1:],
            transform=transform,
            invert=True
        )
        
        pixel_features = feature_array[:, mask].T
        
        pixel_labels = np.full(pixel_features.shape[0], class_label)
        X.append(pixel_features)
        y.append(pixel_labels)
    
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"样本提取完成：总像元样本数{len(X)}，训练集{len(X_train)}，验证集{len(X_test)}")
   
    for class_id, class_name in CLASS_DICT.items():
        count = np.sum(y == class_id)
        print(f"{class_name} 样本像元数：{count}")
    return X_train, X_test, y_train, y_test
# ===================== 5. 随机森林模型训练与精度评价 =====================
def train_random_forest(X_train, X_test, y_train, y_test, feature_names):
   
    print("=" * 60)
    print("开始训练基于像元的随机森林模型...")
   
    rf_model = RandomForestClassifier(**RF_PARAMS)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    
    oa = accuracy_score(y_test, y_pred) * 100
   
    kappa = cohen_kappa_score(y_test, y_pred)
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    class_report = classification_report(
        y_test, y_pred,
        target_names=[CLASS_DICT[i] for i in sorted(CLASS_DICT.keys())],
        output_dict=True
    )
    
    oob_score = rf_model.oob_score_ * 100
    
    print("=" * 60)
    print(f"=== 基于像元的随机森林分类精度结果 ===")
    print(f"总体精度(OA): {oa:.4f}%")
    print(f"Kappa系数: {kappa:.4f}")
    print(f"袋外验证分数(OOB): {oob_score:.4f}%")
    print("=" * 60)
    print("混淆矩阵：")
    print(conf_matrix)
    print("=" * 60)
   
    report_df = pd.DataFrame(class_report).T
    report_df.to_excel(os.path.join(ACCURACY_PATH, "基于像元的随机森林_分类精度报告.xlsx"))
    
    np.savetxt(os.path.join(ACCURACY_PATH, "混淆矩阵.csv"), conf_matrix, delimiter=",", fmt="%d")
   
    feature_importance = pd.DataFrame({
        "特征名称": feature_names,
        "重要性得分": rf_model.feature_importances_
    }).sort_values(by="重要性得分", ascending=False)
    feature_importance.to_excel(os.path.join(ACCURACY_PATH, "特征重要性排序.xlsx"), index=False)
    print("特征重要性排名（Top5）：")
    print(feature_importance.head())
    print("=" * 60)
    return rf_model, oa, kappa
# ===================== 6. 全影像逐像元分类与后处理 =====================
def pixel_wise_classification(rf_model, feature_array, profile, image_name):
   
    print("开始全影像逐像元分类预测...")
   
    band_num, height, width = feature_array.shape
   
    img_reshape = feature_array.reshape(band_num, -1).T
   
    block_size = 200000
    pred_result = np.zeros(img_reshape.shape[0], dtype=np.uint8)
    for i in tqdm(range(0, img_reshape.shape[0], block_size), desc="分块逐像元预测"):
        end = min(i + block_size, img_reshape.shape[0])
        pred_result[i:end] = rf_model.predict(img_reshape[i:end])
    
    pred_img = pred_result.reshape(height, width)
   
    print("执行形态学后处理，去除孤立像元...")
   
    pred_img_filtered = median_filter(pred_img, size=3)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    pred_img_final = cv2.morphologyEx(pred_img_filtered, cv2.MORPH_OPEN, kernel)
    pred_img_final = cv2.morphologyEx(pred_img_final, cv2.MORPH_CLOSE, kernel)
   
    output_profile = profile.copy()
    output_profile.update({
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw",  
        "nodata": 0
    })
   
    raw_output_file = os.path.join(OUTPUT_PATH, f"{image_name}_基于像元的随机森林_原始分类结果.tif")
    with rasterio.open(raw_output_file, "w", **output_profile) as dst:
        dst.write(pred_img, 1)
   
    final_output_file = os.path.join(OUTPUT_PATH, f"{image_name}_基于像元的随机森林_最终分类结果.tif")
    with rasterio.open(final_output_file, "w", **output_profile) as dst:
        dst.write(pred_img_final, 1)
   
    print("=" * 60)
    print(f"分类结果面积统计（像元数）：")
    pixel_area = abs(profile["transform"][0] * profile["transform"][4])  
    for class_id, class_name in CLASS_DICT.items():
        pixel_count = np.sum(pred_img_final == class_id)
        area_sqm = pixel_count * pixel_area
        area_km2 = area_sqm / 1e6
        print(f"{class_name}：像元数{pixel_count}，面积{area_km2:.4f} 平方公里")
    print("=" * 60)
    print(f"原始分类结果已保存至：{raw_output_file}")
    print(f"后处理最终结果已保存至：{final_output_file}")
    return pred_img_final
# ===================== 7. 主函数：执行全流程 =====================
if __name__ == "__main__":
    
    print("读取Sentinel-1 SAR影像...")
    with rasterio.open(IMAGE_PATH) as src:
        img_array = src.read()
        profile = src.profile
        transform = src.transform
        crs = src.crs
    image_name = os.path.basename(IMAGE_PATH).split(".")[0]
    print(f"影像读取完成：极化波段数{img_array.shape[0]}，高度{img_array.shape[1]}，宽度{img_array.shape[2]}")
    
    denoised_img = sar_preprocessing(img_array)
    
    feature_array, feature_names = build_feature_set(denoised_img)
    
    X_train, X_test, y_train, y_test = extract_pixel_samples(feature_array, SAMPLE_SHP_PATH, transform)
    
    rf_model, oa, kappa = train_random_forest(X_train, X_test, y_train, y_test, feature_names)
   
    final_classification = pixel_wise_classification(rf_model, feature_array, profile, image_name)
   
    time_series_dir = r"./sentinel1_time_series/"
   
    image_list = [f for f in os.listdir(time_series_dir) if f.endswith(".tif")]
    
    print(f"开始批量处理{len(image_list)}期时序影像...")
    for image_file in tqdm(image_list, desc="时序批量分类"):
        image_path = os.path.join(time_series_dir, image_file)
        image_name = os.path.basename(image_path).split(".")[0]
        
        
        with rasterio.open(image_path) as src:
            img_array = src.read()
            profile = src.profile
        
       
        denoised_img = sar_preprocessing(img_array)
        feature_array, _ = build_feature_set(denoised_img)
        
       
        pixel_wise_classification(rf_model, feature_array, profile, image_name)
    
    print("2016-2022年时序影像批量分类完成！")
    """
    print("=" * 60)
        print("基于像元的随机森林浮筏养殖区提取全流程执行完成！")
    print("=" * 60)

  # 若浮筏提取结果破碎化严重，可增大形态学滤波的核尺寸（filter_size改为5）;
  # 若模型出现过拟合，可减小n_estimators、增大min_samples_leaf参数;
  # 若需区分更多/更少的浮筏养殖类型，直接修改CLASS_DICT分类字典即可;
  # 针对不同年份的SAR影像，可微调特征集，增加时序归一化步骤，保证年际分类结果的一致性
