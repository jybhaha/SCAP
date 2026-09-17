#!/usr/bin/env python3
"""
实验08：真实世界API测试
测试SCAR对抗电商平台图像识别API的攻击效果

主要功能：
1. 统计每个平台的攻击成功率，以及整体成功率
2. 图片展示：保存攻击图片并标注攻击信息
3. 使用物体类别图片进行测试
"""

import torch
import requests
import json
import time
import base64
import os
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from utils.result_saver import ResultManager
from experiments.experiment_runner import ExperimentRunner
from data.data_loader import get_objects_experiment_dataloader
from configs.experiment_configs import get_config


class RealWorldAPITester:
    """真实世界API测试器"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.rm = ResultManager.get_instance()
        self._init_api_configs()
        
    def _init_api_configs(self):
        """初始化API配置"""
        self.api_configs = {
            "ecommerce": {
                "name": "E-commerce Image Recognition",
                "endpoints": {
                    "taobao": "https://eco.taobao.com/router/rest",
                    "jd": "https://api.jd.com/routerjson",
                    "pinduoduo": "https://open-api.pinduoduo.com/api",
                    "tmall": "https://eco.taobao.com/router/rest"
                },
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer YOUR_API_KEY"
                },
                "timeout": 30
            }
        }
    
    def _image_to_base64(self, image_tensor):
        """将图像张量转换为base64编码"""
        try:
            # 转换为PIL图像
            if image_tensor.dim() == 4:
                image_tensor = image_tensor.squeeze(0)
            
            # 确保值在[0,1]范围内
            image_tensor = torch.clamp(image_tensor, 0, 1)
            
            # 转换为numpy数组
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            
            # 转换为PIL图像
            image_pil = Image.fromarray(image_np)
            
            # 转换为base64
            buffer = BytesIO()
            image_pil.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return image_base64, image_pil
            
        except Exception as e:
            self.rm.log(f"Image conversion failed: {e}")
            return None, None
    
    def _call_api(self, api_type, endpoint_name, image_base64, original_label=None):
        """调用API进行图像识别"""
        try:
            config = self.api_configs[api_type]
            endpoint = config["endpoints"][endpoint_name]
            
            # 构建请求数据
            payload = {
                "image": image_base64,
                "format": "base64"
            }
            
            if original_label is not None:
                payload["expected_label"] = original_label
            
            # 发送请求
            response = requests.post(
                endpoint,
                headers=config["headers"],
                json=payload,
                timeout=config["timeout"]
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "result": result,
                    "response_time": response.elapsed.total_seconds()
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "response_time": response.elapsed.total_seconds()
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timeout",
                "response_time": config["timeout"]
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Request failed: {str(e)}",
                "response_time": 0
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unknown error: {str(e)}",
                "response_time": 0
            }
    
    def _parse_api_response(self, api_type, response):
        """解析API响应 - 改进版本，支持多种响应格式"""
        try:
            if not response["success"]:
                return None, None, response["error"]
            
            result = response["result"]
            self.rm.log(f"API response structure: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            
            # 尝试多种可能的响应格式
            predicted_label = "unknown"
            confidence = 0.0
            
            # 格式1: 直接包含label和confidence
            if "label" in result:
                predicted_label = result["label"]
                confidence = result.get("confidence", 0.0)
            
            # 格式2: 包含categories数组
            elif "categories" in result and isinstance(result["categories"], list) and len(result["categories"]) > 0:
                top_category = result["categories"][0]
                if isinstance(top_category, dict):
                    predicted_label = top_category.get("name", top_category.get("label", "unknown"))
                    confidence = top_category.get("confidence", top_category.get("score", 0.0))
                else:
                    predicted_label = str(top_category)
                    confidence = 0.0
            
            # 格式3: 包含predictions数组
            elif "predictions" in result and isinstance(result["predictions"], list) and len(result["predictions"]) > 0:
                top_prediction = result["predictions"][0]
                if isinstance(top_prediction, dict):
                    predicted_label = top_prediction.get("name", top_prediction.get("label", "unknown"))
                    confidence = top_prediction.get("confidence", top_prediction.get("score", 0.0))
                else:
                    predicted_label = str(top_prediction)
                    confidence = 0.0
            
            # 格式4: 包含class_name和score
            elif "class_name" in result:
                predicted_label = result["class_name"]
                confidence = result.get("score", result.get("confidence", 0.0))
            
            # 格式5: 包含result字段
            elif "result" in result:
                sub_result = result["result"]
                if isinstance(sub_result, dict):
                    predicted_label = sub_result.get("label", sub_result.get("name", "unknown"))
                    confidence = sub_result.get("confidence", sub_result.get("score", 0.0))
                else:
                    predicted_label = str(sub_result)
                    confidence = 0.0
            
            # 格式6: 如果result是字符串，直接使用
            elif isinstance(result, str):
                predicted_label = result
                confidence = 0.0
            
            # 格式7: 如果result是列表，取第一个元素
            elif isinstance(result, list) and len(result) > 0:
                first_item = result[0]
                if isinstance(first_item, dict):
                    predicted_label = first_item.get("label", first_item.get("name", "unknown"))
                    confidence = first_item.get("confidence", first_item.get("score", 0.0))
                else:
                    predicted_label = str(first_item)
                    confidence = 0.0
            
            # 如果还是unknown，尝试从所有字段中查找
            if predicted_label == "unknown":
                for key, value in result.items():
                    if key.lower() in ["label", "name", "class", "category", "prediction"]:
                        if isinstance(value, str):
                            predicted_label = value
                            break
                        elif isinstance(value, list) and len(value) > 0:
                            predicted_label = str(value[0])
                            break
            
            # 确保confidence是数值类型
            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                confidence = 0.0
            
            self.rm.log(f"Parsed result: label='{predicted_label}', confidence={confidence}")
            return predicted_label, confidence, None
            
        except Exception as e:
            self.rm.log(f"Response parsing failed: {str(e)}")
            return None, None, f"Response parsing failed: {str(e)}"
    
    def _create_annotated_image(self, image_pil, api_name, original_label, predicted_label, 
                               confidence, attack_success, sample_idx, class_names):
        """创建带标注的攻击图片"""
        try:
            # 创建副本
            annotated_img = image_pil.copy()
            draw = ImageDraw.Draw(annotated_img)
            
            # 设置字体（尝试使用系统字体）
            try:
                font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
                font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except:
                try:
                    font_large = ImageFont.load_default()
                    font_medium = ImageFont.load_default()
                    font_small = ImageFont.load_default()
                except:
                    font_large = None
                    font_medium = None
                    font_small = None
            
            # 获取图片尺寸
            width, height = annotated_img.size
            
            # 创建几乎全透明的背景
            overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            # 绘制信息背景框（几乎全透明）
            info_height = 140  # 增加高度以容纳更多信息
            overlay_draw.rectangle([0, height - info_height, width, height], 
                                 fill=(0, 0, 0, 30))  # 几乎全透明
            
            # 合并图片
            annotated_img = Image.alpha_composite(annotated_img.convert('RGBA'), overlay)
            draw = ImageDraw.Draw(annotated_img)
            
            # 绘制文字信息
            y_offset = height - info_height + 10
            
            # 平台名称（使用英文全称）
            platform_full_names = {
                "taobao": "TAOBAO",
                "jd": "JD.COM", 
                "pinduoduo": "PINDUODUO",
                "tmall": "TMALL"
            }
            platform_text = f"Platform: {platform_full_names.get(api_name, api_name.upper())}"
            draw.text((10, y_offset), platform_text, fill=(255, 255, 255), font=font_large)
            y_offset += 25
            
            # 原始标签（显示类别名称）
            original_class_name = class_names[original_label] if original_label < len(class_names) else f"class_{original_label}"
            original_text = f"Original: {original_class_name}"
            draw.text((10, y_offset), original_text, fill=(255, 255, 255), font=font_medium)
            y_offset += 20
            
            # 预测标签
            predicted_text = f"Predicted: {predicted_label}"
            draw.text((10, y_offset), predicted_text, fill=(255, 255, 255), font=font_medium)
            y_offset += 20
            
            # 置信度
            confidence_text = f"Confidence: {confidence:.3f}"
            draw.text((10, y_offset), confidence_text, fill=(255, 255, 255), font=font_medium)
            y_offset += 20
            
            # 攻击结果
            if attack_success:
                result_text = "Attack: SUCCESS"
                result_color = (255, 100, 100)  # 红色
            else:
                result_text = "Attack: FAILED"
                result_color = (100, 255, 100)  # 绿色
            
            draw.text((10, y_offset), result_text, fill=result_color, font=font_large)
            y_offset += 25
            
            # 在右上角添加编号
            number_text = f"#{sample_idx}"
            text_bbox = draw.textbbox((0, 0), number_text, font=font_small)
            text_width = text_bbox[2] - text_bbox[0]
            draw.text((width - text_width - 10, 10), number_text, 
                     fill=(255, 255, 255), font=font_small)
            
            return annotated_img
            
        except Exception as e:
            self.rm.log(f"Failed to create annotated image: {e}")
            return image_pil
    
    def test_ecommerce_apis(self, dataloader, class_names, config):
        """测试电商平台图像识别API"""
        self.rm.log("Starting e-commerce API testing with real API calls...")
        
        # 创建结果保存目录
        result_dir = "Results/experiment_08_api_test"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        # 为每个平台创建子目录（使用英文全称）
        platform_full_names = {
            "taobao": "TAOBAO",
            "jd": "JD_COM", 
            "pinduoduo": "PINDUODUO",
            "tmall": "TMALL"
        }
        
        for api_name in self.api_configs["ecommerce"]["endpoints"].keys():
            platform_name = platform_full_names.get(api_name, api_name.upper())
            api_dir = os.path.join(result_dir, platform_name)
            if not os.path.exists(api_dir):
                os.makedirs(api_dir)
        
        results = {
            "apis": {},
            "overall_asr": 0.0,
            "total_samples": 0,
            "successful_attacks": 0,
            "class_names": class_names
        }
        
        total_samples = 0
        successful_attacks = 0
        sample_idx = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            for i in range(images.size(0)):
                img = images[i:i+1]
                label = labels[i:i+1]
                sample_idx += 1
                
                # 转换图像为base64和PIL图像
                image_base64, image_pil = self._image_to_base64(img)
                if image_base64 is None or image_pil is None:
                    continue
                
                # 测试每个API
                for api_name in self.api_configs["ecommerce"]["endpoints"].keys():
                    if api_name not in results["apis"]:
                        results["apis"][api_name] = {
                            "asr": 0.0,
                            "successful_attacks": 0,
                            "total_samples": 0,
                            "response_times": [],
                            "errors": []
                        }
                    
                    # 调用真实API
                    self.rm.log(f"Calling {api_name} API for sample {sample_idx}...")
                    response = self._call_api("ecommerce", api_name, image_base64, label.item())
                    
                    # 解析响应
                    predicted_label, confidence, error = self._parse_api_response("ecommerce", response)
                    
                    if error:
                        self.rm.log(f"API call failed for {api_name}: {error}")
                        results["apis"][api_name]["errors"].append(error)
                        # 保存错误情况的图片
                        error_img = self._create_annotated_image(
                            image_pil, api_name, label.item(), "ERROR", 0.0, False, sample_idx, class_names
                        )
                        platform_name = platform_full_names.get(api_name, api_name.upper())
                        error_img.save(os.path.join(result_dir, platform_name, f"error_{sample_idx}.png"))
                        continue
                    
                    # 记录响应时间
                    results["apis"][api_name]["response_times"].append(response["response_time"])
                    
                    # 判断攻击是否成功（预测标签与原始标签不同）
                    original_class_name = class_names[label.item()] if label.item() < len(class_names) else f"class_{label.item()}"
                    attack_success = predicted_label != original_class_name
                    if attack_success:
                        results["apis"][api_name]["successful_attacks"] += 1
                        successful_attacks += 1
                    
                    results["apis"][api_name]["total_samples"] += 1
                    total_samples += 1
                    
                    # 创建带标注的图片
                    annotated_img = self._create_annotated_image(
                        image_pil, api_name, label.item(), predicted_label, 
                        confidence, attack_success, sample_idx, class_names
                    )
                    
                    # 保存图片（使用英文全称目录名）
                    platform_name = platform_full_names.get(api_name, api_name.upper())
                    success_status = "SUCCESS" if attack_success else "FAILED"
                    class_name = class_names[label.item()] if label.item() < len(class_names) else f"class_{label.item()}"
                    filename = f"{sample_idx}_{class_name}_{success_status}_conf_{confidence:.3f}.png"
                    annotated_img.save(os.path.join(result_dir, platform_name, filename))
                    
                    self.rm.log(f"{api_name} API result: {predicted_label} (confidence: {confidence:.3f}, attack: {'SUCCESS' if attack_success else 'FAILED'})")
                
                if sample_idx % 10 == 0:
                    self.rm.log(f"Processed {sample_idx} samples")
        
        # 计算ASR
        for api_name in results["apis"]:
            api_data = results["apis"][api_name]
            if api_data["total_samples"] > 0:
                api_data["asr"] = api_data["successful_attacks"] / api_data["total_samples"]
        
        results["overall_asr"] = successful_attacks / total_samples if total_samples > 0 else 0.0
        results["total_samples"] = total_samples
        results["successful_attacks"] = successful_attacks
        
        # 保存统计结果
        self._save_statistics(results, result_dir)
        
        return results
    
    def _save_statistics(self, results, result_dir):
        """保存统计结果"""
        # 保存JSON格式的详细结果
        with open(os.path.join(result_dir, "statistics.json"), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存Markdown格式的统计报告
        md_report = self._generate_markdown_report(results)
        with open(os.path.join(result_dir, "statistics_report.md"), 'w', encoding='utf-8') as f:
            f.write(md_report)
        
        # 保存LaTeX格式的表格
        latex_table = self._generate_latex_table(results)
        with open(os.path.join(result_dir, "statistics_table.tex"), 'w', encoding='utf-8') as f:
            f.write(latex_table)
    
    def _generate_markdown_report(self, results):
        """生成Markdown格式的统计报告"""
        report = "# E-commerce API Attack Test Results Report (Real API Calls)\n\n"
        report += f"**Test Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## Overall Statistics\n\n"
        report += f"- **Total Samples**: {results['total_samples']}\n"
        report += f"- **Successful Attacks**: {results['successful_attacks']}\n"
        report += f"- **Overall Attack Success Rate**: {results['overall_asr']:.2%}\n\n"
        
        report += "## Object Classes Tested\n\n"
        report += f"Classes: {', '.join(results['class_names'])}\n\n"
        
        report += "## Platform Detailed Results\n\n"
        report += "| Platform | ASR | Successful Attacks | Total Samples | Avg Response Time(s) | Error Count |\n"
        report += "|----------|-----|-------------------|---------------|---------------------|-------------|\n"
        
        # 平台全称映射
        platform_full_names = {
            "taobao": "TAOBAO",
            "jd": "JD.COM", 
            "pinduoduo": "PINDUODUO",
            "tmall": "TMALL"
        }
        
        for api_name, api_data in results["apis"].items():
            platform_name = platform_full_names.get(api_name, api_name.upper())
            avg_time = np.mean(api_data["response_times"]) if api_data["response_times"] else 0
            error_count = len(api_data["errors"])
            report += f"| {platform_name} | {api_data['asr']:.2%} | {api_data['successful_attacks']} | {api_data['total_samples']} | {avg_time:.3f} | {error_count} |\n"
        
        report += f"| **Overall** | **{results['overall_asr']:.2%}** | **{results['successful_attacks']}** | **{results['total_samples']}** | - | - |\n\n"
        
        # 添加错误信息
        if any(len(api_data["errors"]) > 0 for api_data in results["apis"].values()):
            report += "## API Error Information\n\n"
            for api_name, api_data in results["apis"].items():
                if api_data["errors"]:
                    platform_name = platform_full_names.get(api_name, api_name.upper())
                    report += f"### {platform_name} Errors\n\n"
                    for i, error in enumerate(api_data["errors"][:5]):  # 只显示前5个错误
                        report += f"{i+1}. {error}\n"
                    if len(api_data["errors"]) > 5:
                        report += f"... and {len(api_data['errors']) - 5} more errors\n"
                    report += "\n"
        
        report += "## Results Analysis\n\n"
        if results['overall_asr'] > 0.8:
            report += "✅ **Excellent**: SCAR performs excellently on e-commerce APIs with high attack success rate\n"
        elif results['overall_asr'] > 0.5:
            report += "⚠️ **Good**: SCAR performs well on e-commerce APIs but has room for improvement\n"
        else:
            report += "❌ **Average**: SCAR performs averagely on e-commerce APIs and needs further optimization\n"
        
        return report
    
    def _generate_latex_table(self, results):
        """生成LaTeX格式的统计表格"""
        latex = """\\begin{table}[htbp]
\\centering
\\caption{Attack Success Rate Results on E-commerce APIs (Real API Calls)}
\\label{tab:asr_results_real_apis}
\\begin{tabular}{lccccc}
\\toprule
Platform & ASR & Successful Attacks & Total Samples & Avg Response Time (s) & Errors \\\\
\\midrule
"""
        
        # 平台全称映射
        platform_full_names = {
            "taobao": "TAOBAO",
            "jd": "JD.COM", 
            "pinduoduo": "PINDUODUO",
            "tmall": "TMALL"
        }
        
        for api_name, api_data in results["apis"].items():
            platform_name = platform_full_names.get(api_name, api_name.upper())
            asr = api_data["asr"]
            success = api_data["successful_attacks"]
            total = api_data["total_samples"]
            avg_time = np.mean(api_data["response_times"]) if api_data["response_times"] else 0
            error_count = len(api_data["errors"])
            
            latex += f"{platform_name} & {asr:.2%} & {success} & {total} & {avg_time:.3f} & {error_count} \\\\\n"
        
        latex += f"""\\midrule
\\textbf{{Overall}} & \\textbf{{{results['overall_asr']:.2%}}} & \\textbf{{{results['successful_attacks']}}} & \\textbf{{{results['total_samples']}}} & - & - \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
        
        return latex


def run_real_world_api_test_experiment(dataloader_mode="small", categories="household"):
    """
    运行真实世界API测试实验
    Args:
        dataloader_mode: 数据加载器模式
        categories: 物体类别，可选 "household", "tools", "vehicles", "sports" 或组合
    """
    rm = ResultManager.get_instance()
    rm.set_experiment("RealWorldAPITest")
    rm.set_test("api_test_real_apis")
    
    # 获取物体类别数据加载器
    rm.log(f"Loading object images from category: {categories}")
    try:
        dataloader, subset, class_to_idx = get_objects_experiment_dataloader(
            experiment_type=dataloader_mode, 
            categories=categories
        )
        class_names = list(class_to_idx.keys())
        rm.log(f"Loaded {len(class_names)} object classes: {class_names}")
        
        # 验证数据加载器是否包含物体图片
        rm.log("Verifying dataloader contains object images...")
        sample_count = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            sample_count += images.size(0)
            if batch_idx == 0:  # 只检查第一个batch
                rm.log(f"First batch: {images.size(0)} images, labels: {labels.tolist()}")
                rm.log(f"Label range: {labels.min().item()} to {labels.max().item()}")
                rm.log(f"Class names for first batch: {[class_names[i] for i in labels.tolist()]}")
            if sample_count >= 20:  # 检查前20个样本
                break
        
        rm.log(f"Verified {sample_count} samples from object categories")
        
    except Exception as e:
        rm.log(f"Failed to load object dataloader: {e}")
        rm.log("Falling back to regular dataloader...")
        from data.data_loader import get_experiment_dataloader
        dataloader = get_experiment_dataloader(dataloader_mode)
        class_names = [f"class_{i}" for i in range(1000)]  # 默认ImageNet类别
    
    # 获取配置
    config = get_config("default")
    
    # 初始化API测试器
    api_tester = RealWorldAPITester()
    
    # 测试电商平台图像识别API
    rm.log("=== Starting E-commerce API Testing with Real API Calls ===")
    ecommerce_results = api_tester.test_ecommerce_apis(dataloader, class_names, config)
    rm.log(f"E-commerce API testing completed, Overall ASR: {ecommerce_results['overall_asr']:.2%}")
    
    # 输出结果摘要
    rm.log("=== E-commerce API Test Results Summary ===")
    platform_full_names = {
        "taobao": "TAOBAO",
        "jd": "JD.COM", 
        "pinduoduo": "PINDUODUO",
        "tmall": "TMALL"
    }
    
    for api_name, api_data in ecommerce_results["apis"].items():
        platform_name = platform_full_names.get(api_name, api_name.upper())
        error_count = len(api_data["errors"])
        rm.log(f"{platform_name}: ASR = {api_data['asr']:.2%} ({api_data['successful_attacks']}/{api_data['total_samples']}), Errors = {error_count}")
    
    rm.log(f"Overall ASR: {ecommerce_results['overall_asr']:.2%} ({ecommerce_results['successful_attacks']}/{ecommerce_results['total_samples']})")
    rm.log(f"Object classes tested: {', '.join(class_names)}")
    
    rm.log("Real-world API test experiment completed!")
    rm.log(f"Results and images saved to: Results/experiment_08_api_test/")
    
    return ecommerce_results


if __name__ == '__main__':
    # 可以指定不同的物体类别进行测试
    run_real_world_api_test_experiment(dataloader_mode="small", categories="household")
    # run_real_world_api_test_experiment(dataloader_mode="small", categories=["household", "tools"])
    # run_real_world_api_test_experiment(dataloader_mode="small", categories="vehicles") 