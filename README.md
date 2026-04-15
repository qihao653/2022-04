# 2022-04
基于像元的随机森林方法提取浮筏养殖区
pip install rasterio numpy scikit-learn matplotlib pandas geopandas tqdm scipy opencv-python scikit-image
# ===================== 江苏近海浮筏养殖区 基于像元的随机森林分类代码 =====================
# 适配大创项目：基于SAR影像的海上浮筏养殖信息提取与变化检测研究
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
# 1.1 分类体系与编码（项目中区分3类浮筏养殖+背景，可按需调整）
CLASS_DICT = {
    0: "海水背景",
    1: "浮筏养殖-紫菜筏式养殖",
    2: "浮筏养殖-贝类筏式养殖",
    3: "浮筏养殖-网箱养殖",
    4: "岸滩/陆地非养殖区"
}
CLASS_NUM = len(CLASS_DICT) - 1  # 分类类别数
# 1.2 随机森林模型核心参数（与项目实验参数一致）
RF_PARAMS = {
    "n_estimators": 150,  # 决策树数量，适配SAR影像细碎特征
    "criterion": "gini",  # 基尼系数分裂准则
    "max_depth": 12,  # 限制树深度，防止过拟合
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": 42,  # 固定随机种子，结果可复现
    "n_jobs": -1,  # 调用全部CPU核心加速
    "oob_score": True,  # 袋外分数验证模型泛化能力
    "verbose": 1
}
# 1.3 SAR影像波段配置（Sentinel-1 双极化数据，预处理后波段顺序：VV、VH）
BAND_NAMES = ["VV", "VH"]
BAND_INDEX = {name: i for i, name in enumerate(BAND_NAMES)}
# 1.4 文件路径配置（根据你的本地文件修改）
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
    """
    SAR影像预处理：Lee滤波抑制相干斑噪声，对应项目中相干斑噪声影响问题
    :param img_array: 输入SAR影像数组，shape=(波段数, 高度, 宽度)
    :param filter_size: 滤波窗口大小
    :return: 去噪后的影像数组
    """
    print("开始SAR影像相干斑噪声抑制...")
    denoised_array = np.zeros_like(img_array, dtype=np.float32)
    
    # 对每个极化波段做Lee滤波
    for i in range(img_array.shape[0]):
        band_data = img_array[i, :, :]
        # 均值滤波
        mean = cv2.blur(band_data, (filter_size, filter_size))
        # 方差滤波
        var = cv2.blur(band_data ** 2, (filter_size, filter_size)) - mean ** 2
        # 噪声方差
        noise_var = np.mean(var)
        # Lee滤波权重
        weight = var / (var + noise_var + 1e-8)
        denoised_band = mean + weight * (band_data - mean)
        denoised_array[i, :, :] = denoised_band
    
    # 处理异常值
    denoised_array = np.nan_to_num(denoised_array, nan=0, posinf=0, neginf=0)
    print("SAR影像预处理完成")
    return denoised_array
# ===================== 3. 特征集构建（复现项目中12维特征，极化+水体指数+纹理+极化分解）=====================
def build_feature_set(img_array):
    """
    构建项目中提到的12维浮筏养殖提取特征集，适配SAR影像浮筏目标特性
    :param img_array: 去噪后的SAR影像数组，shape=(2, 高度, 宽度)
    :return: 特征数组，shape=(12, 高度, 宽度)，特征名称列表
    """
    print("开始构建12维分类特征集...")
    # 提取双极化波段
    VV = img_array[BAND_INDEX["VV"], :, :].astype(np.float32)
    VH = img_array[BAND_INDEX["VH"], :, :].astype(np.float32)
    epsilon = 1e-8  # 避免分母为0
    # ---------------------- 第一类：基础极化特征（5个）----------------------
    # 1. VV极化后向散射系数
    # 2. VH极化后向散射系数
    # 3. VV/VH极化比值（浮筏与海水的极化差异核心特征）
    VV_VH_ratio = VV / (VH + epsilon)
    # 4. VH/VV极化比值
    VH_VV_ratio = VH / (VV + epsilon)
    # 5. VV-VH极化差值
    VV_VH_diff = VV - VH
    # ---------------------- 第二类：SAR指数特征（2个）----------------------
    # 6. 雷达植被指数RVI（适配浮筏养殖植被/人工结构散射特性）
    RVI = (4 * VH) / (VV + VH + epsilon)
    # 7. SAR水体指数SWI（区分海水背景与浮筏目标）
    SWI = (VH - VV) / (VH + VV + epsilon)
    # ---------------------- 第三类：GLCM纹理特征（4个）----------------------
    # 对VH波段（浮筏特征更显著）计算灰度共生矩阵纹理特征
    # 先将后向散射系数归一化到0-255灰度级
    vh_normalized = cv2.normalize(VH, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 计算GLCM（步长1，方向0°）
    glcm = graycomatrix(vh_normalized, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    # 8. 对比度（Contrast）：浮筏与海水边界差异
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    # 9. 熵（Entropy）：纹理复杂度，浮筏区域纹理更复杂
    entropy = -np.sum(glcm * np.log2(glcm + epsilon))
    # 10. 相关性（Correlation）：纹理线性相关性
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    # 11. 二阶矩（ASM）：纹理均匀性
    asm = graycoprops(glcm, 'ASM')[0, 0]
    # ---------------------- 第四类：极化分解特征（1个）----------------------
    # 12. 散射熵（双极化近似）：区分浮筏的二次散射与海水的面散射
    scatter_entropy = - (VV * np.log2(VV + epsilon) + VH * np.log2(VH + epsilon)) / 2
    # 合并所有12维特征
    feature_list = [
        VV, VH, VV_VH_ratio, VH_VV_ratio, VV_VH_diff,
        RVI, SWI, contrast, entropy, correlation, asm, scatter_entropy
    ]
    feature_array = np.stack(feature_list, axis=0)
    feature_names = [
        "VV", "VH", "VV/VH", "VH/VV", "VV-VH",
        "RVI", "SWI", "Contrast", "Entropy", "Correlation", "ASM", "Scatter_Entropy"
    ]
    # 处理异常值
    feature_array = np.nan_to_num(feature_array, nan=0, posinf=0, neginf=0)
    print(f"特征集构建完成，共{len(feature_names)}维特征")
    return feature_array, feature_names
# ===================== 4. 基于像元的样本特征提取 =====================
def extract_pixel_samples(feature_array, sample_shp, transform):
    """
    逐像元提取ROI样本的特征与标签，对应项目中「基于像元的随机森林」核心逻辑
    :param feature_array: 12维特征数组
    :param sample_shp: 样本矢量shp路径
    :param transform: 影像地理变换参数
    :return: 训练集、验证集的特征与标签
    """
    print("开始逐像元提取训练样本特征...")
    # 读取样本矢量
    gdf = gpd.read_file(sample_shp)
    # 校验class字段
    if "class" not in gdf.columns:
        raise ValueError("样本shp文件必须包含'class'字段，对应分类编码！")
    # 初始化特征与标签列表
    X = []
    y = []
    # 遍历每个样本多边形，逐像元提取特征
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="提取像元样本"):
        class_label = row["class"]
        geometry = row["geometry"]
        # 生成多边形掩膜，获取对应像元位置
        mask = geometry_mask(
            [geometry],
            out_shape=feature_array.shape[1:],
            transform=transform,
            invert=True
        )
        # 提取掩膜内的逐像元特征
        pixel_features = feature_array[:, mask].T
        # 对应标签
        pixel_labels = np.full(pixel_features.shape[0], class_label)
        X.append(pixel_features)
        y.append(pixel_labels)
    # 合并所有样本
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    # 划分训练集(70%)和验证集(30%)，分层抽样保证类别均衡
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"样本提取完成：总像元样本数{len(X)}，训练集{len(X_train)}，验证集{len(X_test)}")
    # 输出各类别样本数量
    for class_id, class_name in CLASS_DICT.items():
        count = np.sum(y == class_id)
        print(f"{class_name} 样本像元数：{count}")
    return X_train, X_test, y_train, y_test
# ===================== 5. 随机森林模型训练与精度评价 =====================
def train_random_forest(X_train, X_test, y_train, y_test, feature_names):
    """
    训练基于像元的随机森林模型，输出项目要求的全维度精度评价指标
    :return: 训练好的模型、核心精度指标
    """
    print("=" * 60)
    print("开始训练基于像元的随机森林模型...")
    # 初始化并训练模型
    rf_model = RandomForestClassifier(**RF_PARAMS)
    rf_model.fit(X_train, y_train)
    # 验证集预测
    y_pred = rf_model.predict(X_test)
    # ---------------------- 精度计算（与项目精度评价体系完全匹配）----------------------
    # 1. 总体精度OA
    oa = accuracy_score(y_test, y_pred) * 100
    # 2. Kappa系数
    kappa = cohen_kappa_score(y_test, y_pred)
    # 3. 混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    # 4. 分类报告（制图精度/用户精度、F1分数、召回率、精确率）
    class_report = classification_report(
        y_test, y_pred,
        target_names=[CLASS_DICT[i] for i in sorted(CLASS_DICT.keys())],
        output_dict=True
    )
    # 5. 袋外分数OOB
    oob_score = rf_model.oob_score_ * 100
    # ---------------------- 精度结果输出 ----------------------
    print("=" * 60)
    print(f"=== 基于像元的随机森林分类精度结果 ===")
    print(f"总体精度(OA): {oa:.4f}%")
    print(f"Kappa系数: {kappa:.4f}")
    print(f"袋外验证分数(OOB): {oob_score:.4f}%")
    print("=" * 60)
    print("混淆矩阵：")
    print(conf_matrix)
    print("=" * 60)
    # 保存精度结果到本地
    # 分类报告
    report_df = pd.DataFrame(class_report).T
    report_df.to_excel(os.path.join(ACCURACY_PATH, "基于像元的随机森林_分类精度报告.xlsx"))
    # 混淆矩阵
    np.savetxt(os.path.join(ACCURACY_PATH, "混淆矩阵.csv"), conf_matrix, delimiter=",", fmt="%d")
    # 特征重要性（分析各特征对浮筏提取的贡献度）
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
    """
    对整幅SAR影像进行逐像元分类，对应项目中「基于像元的随机森林」核心流程
    增加形态学后处理，解决项目中提到的「提取结果破碎、孤立像元多」的问题
    """
    print("开始全影像逐像元分类预测...")
    # 获取影像维度
    band_num, height, width = feature_array.shape
    # 重塑为模型输入格式（像元数×特征数）
    img_reshape = feature_array.reshape(band_num, -1).T
    # 分块预测，避免大影像内存溢出
    block_size = 200000
    pred_result = np.zeros(img_reshape.shape[0], dtype=np.uint8)
    for i in tqdm(range(0, img_reshape.shape[0], block_size), desc="分块逐像元预测"):
        end = min(i + block_size, img_reshape.shape[0])
        pred_result[i:end] = rf_model.predict(img_reshape[i:end])
    # 重塑回影像二维维度
    pred_img = pred_result.reshape(height, width)
    # ---------------------- 形态学后处理，解决破碎化问题 ----------------------
    print("执行形态学后处理，去除孤立像元...")
    # 3×3中值滤波去除椒盐噪声
    pred_img_filtered = median_filter(pred_img, size=3)
    # 开运算去除小斑点，闭运算填充孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    pred_img_final = cv2.morphologyEx(pred_img_filtered, cv2.MORPH_OPEN, kernel)
    pred_img_final = cv2.morphologyEx(pred_img_final, cv2.MORPH_CLOSE, kernel)
    # ---------------------- 分类结果保存 ----------------------
    output_profile = profile.copy()
    output_profile.update({
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw",  # LZW压缩，减小文件体积
        "nodata": 0
    })
    # 保存原始分类结果
    raw_output_file = os.path.join(OUTPUT_PATH, f"{image_name}_基于像元的随机森林_原始分类结果.tif")
    with rasterio.open(raw_output_file, "w", **output_profile) as dst:
        dst.write(pred_img, 1)
    # 保存后处理最终结果
    final_output_file = os.path.join(OUTPUT_PATH, f"{image_name}_基于像元的随机森林_最终分类结果.tif")
    with rasterio.open(final_output_file, "w", **output_profile) as dst:
        dst.write(pred_img_final, 1)
    # 统计各类别面积
    print("=" * 60)
    print(f"分类结果面积统计（像元数）：")
    pixel_area = abs(profile["transform"][0] * profile["transform"][4])  # 单个像元面积（平方米）
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
    # 1. 读取SAR影像
    print("读取Sentinel-1 SAR影像...")
    with rasterio.open(IMAGE_PATH) as src:
        img_array = src.read()
        profile = src.profile
        transform = src.transform
        crs = src.crs
    image_name = os.path.basename(IMAGE_PATH).split(".")[0]
    print(f"影像读取完成：极化波段数{img_array.shape[0]}，高度{img_array.shape[1]}，宽度{img_array.shape[2]}")
    # 2. SAR影像预处理（相干斑去噪）
    denoised_img = sar_preprocessing(img_array)
    # 3. 构建12维分类特征集
    feature_array, feature_names = build_feature_set(denoised_img)
    # 4. 提取逐像元训练/验证样本
    X_train, X_test, y_train, y_test = extract_pixel_samples(feature_array, SAMPLE_SHP_PATH, transform)
    # 5. 训练随机森林模型，输出精度评价
    rf_model, oa, kappa = train_random_forest(X_train, X_test, y_train, y_test, feature_names)
    # 6. 全影像逐像元分类与后处理
    final_classification = pixel_wise_classification(rf_model, feature_array, profile, image_name)
    # ---------------------- 可选：2016-2022年时序影像批量分类 ----------------------
    # 取消下方注释，即可一键批量处理7年时序影像，匹配项目中动态变化检测需求
    """
    # 时序影像文件夹路径
    time_series_dir = r"./sentinel1_time_series/"
    # 获取所有tif影像
    image_list = [f for f in os.listdir(time_series_dir) if f.endswith(".tif")]
    
    print(f"开始批量处理{len(image_list)}期时序影像...")
    for image_file in tqdm(image_list, desc="时序批量分类"):
        image_path = os.path.join(time_series_dir, image_file)
        image_name = os.path.basename(image_path).split(".")[0]
        
        # 读取影像
        with rasterio.open(image_path) as src:
            img_array = src.read()
            profile = src.profile
        
        # 预处理+特征构建
        denoised_img = sar_preprocessing(img_array)
        feature_array, _ = build_feature_set(denoised_img)
        
        # 逐像元分类
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
