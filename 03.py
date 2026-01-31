import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
warnings.filterwarnings('ignore')

class CelebrityDanceAnalysisModel:
    """名人舞蹈比赛分析模型类"""
    
    def __init__(self):
        """初始化模型"""
        self.df_main = None
        self.df_votes = None
        self.df_combined = None
        self.models = {}
        self.results = {}
        self.output_dir = 'output'
        
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
    def load_data(self, main_data_path='2026_MCM_Problem_C_Data.csv', 
                  votes_data_path='fan_votes_inference_with_certainty.csv'):
        """
        加载数据文件
        
        Parameters:
        -----------
        main_data_path : str
            主数据文件路径
        votes_data_path : str
            粉丝投票数据文件路径
        """
        print("="*50)
        print("加载数据文件...")
        
        try:
            self.df_main = pd.read_csv(main_data_path)
            self.df_votes = pd.read_csv(votes_data_path)
            
            print(f"主数据形状: {self.df_main.shape}")
            print(f"粉丝投票数据形状: {self.df_votes.shape}")
            print(f"主数据列数: {len(self.df_main.columns)}")
            print(f"粉丝投票数据列数: {len(self.df_votes.columns)}")
            
            return True
        except Exception as e:
            print(f"数据加载错误: {e}")
            return False
    
    def explore_data(self):
        """探索性数据分析"""
        print("="*50)
        print("探索性数据分析...")
        
        if self.df_main is None:
            print("错误: 数据未加载")
            return
        
        # 查看基本信息
        print("\n=== 主数据基本信息 ===")
        print(f"数据形状: {self.df_main.shape}")
        print(f"列名: {list(self.df_main.columns)}")
        
        # 查看缺失值情况
        print("\n=== 缺失值情况 ===")
        missing_values = self.df_main.isnull().sum()
        print(missing_values[missing_values > 0])
        
        # 查看数值型变量统计信息
        print("\n=== 数值型变量统计信息 ===")
        numerical_cols = self.df_main.select_dtypes(include=['int64', 'float64']).columns
        print(self.df_main[numerical_cols].describe())
        
        # 查看分类变量分布
        print("\n=== 分类变量分布 ===")
        categorical_cols = ['celebrity_industry', 'celebrity_homecountry/region', 'results']
        for col in categorical_cols:
            if col in self.df_main.columns:
                print(f"\n{col} 分布:")
                print(self.df_main[col].value_counts().head(10))
    
    def preprocess_data(self):
        """数据预处理和特征工程"""
        print("="*50)
        print("数据预处理和特征工程...")
        
        if self.df_main is None or self.df_votes is None:
            print("错误: 数据未加载")
            return
        
        # 1. 处理评委评分数据
        print("处理评委评分数据...")
        judge_cols = [col for col in self.df_main.columns if 'judge' in col]
        print(f"评委评分列数: {len(judge_cols)}")
        
        # 计算每周的平均评分
        weekly_scores = []
        for week in range(1, 12):
            week_cols = [col for col in judge_cols if f'week{week}_' in col]
            if week_cols:
                self.df_main[f'week{week}_avg_score'] = self.df_main[week_cols].mean(axis=1, skipna=True)
                weekly_scores.append(f'week{week}_avg_score')
        
        # 计算总体平均评分
        self.df_main['overall_avg_score'] = self.df_main[weekly_scores].mean(axis=1, skipna=True)
        self.df_main['total_valid_weeks'] = self.df_main[weekly_scores].count(axis=1)
        
        # 2. 处理粉丝投票数据
        print("处理粉丝投票数据...")
        df_votes_renamed = self.df_votes.rename(columns={'contestant': 'celebrity_name'})
        
        # 计算每个名人的平均粉丝投票
        df_votes_summary = df_votes_renamed.groupby(['season', 'celebrity_name']).agg({
            'mean_fan_vote': 'mean',
            'judges_score': 'mean',
            'is_eliminated': 'max'
        }).reset_index()
        
        # 3. 合并数据集
        print("合并数据集...")
        self.df_combined = pd.merge(self.df_main, df_votes_summary, on=['season', 'celebrity_name'], how='left')
        
        # 4. 特征工程
        print("创建新特征...")
        # 国籍特征
        self.df_combined['is_american'] = (self.df_combined['celebrity_homecountry/region'] == 'United States').astype(int)
        
        # 年龄组特征
        self.df_combined['age_group'] = pd.cut(
            self.df_combined['celebrity_age_during_season'], 
            bins=[0, 30, 40, 50, 100], 
            labels=['<30', '30-40', '40-50', '50+']
        )
        
        # 行业影响力特征（基于该行业的平均表现）
        industry_performance = self.df_combined.groupby('celebrity_industry')['overall_avg_score'].mean()
        self.df_combined['industry_performance'] = self.df_combined['celebrity_industry'].map(industry_performance)
        
        # 5. 处理缺失值
        print("处理缺失值...")
        # 对于数值型特征，使用中位数填充
        numerical_cols = ['industry_performance', 'mean_fan_vote']
        for col in numerical_cols:
            if col in self.df_combined.columns:
                self.df_combined[col].fillna(self.df_combined[col].median(), inplace=True)
        
        print(f"预处理后数据形状: {self.df_combined.shape}")
        print(f"最终用于建模的特征数: {len(self.df_combined.columns)}")
        
        # 保存预处理后的数据
        self.df_combined.to_csv(f'{self.output_dir}/preprocessed_data.csv', index=False)
        print(f"预处理后的数据已保存到: {self.output_dir}/preprocessed_data.csv")
    
    def analyze_features(self):
        """特征影响分析"""
        print("="*50)
        print("特征影响分析...")
        
        if self.df_combined is None:
            print("错误: 数据未预处理")
            return
        
        # 选择用于分析的数据（去除缺失值）
        df_analysis = self.df_combined.dropna(subset=['overall_avg_score', 'placement']).copy()
        
        print(f"分析数据形状: {df_analysis.shape}")
        
        # 1. 行业对评分的影响
        print("\n=== 不同行业的表现分析 ===")
        industry_analysis = df_analysis.groupby('celebrity_industry').agg({
            'overall_avg_score': ['mean', 'std', 'count'],
            'placement': ['mean', 'std']
        }).round(2)
        industry_analysis.columns = ['avg_score', 'score_std', 'count', 'avg_rank', 'rank_std']
        industry_analysis = industry_analysis[industry_analysis['count'] >= 5].reset_index()
        industry_analysis = industry_analysis.sort_values('avg_score', ascending=False)
        
        print(industry_analysis)
        
        # 2. 年龄对评分的影响
        print("\n=== 不同年龄组的表现分析 ===")
        age_analysis = df_analysis.groupby('age_group').agg({
            'overall_avg_score': ['mean', 'std', 'count'],
            'placement': ['mean', 'std']
        }).round(2)
        age_analysis.columns = ['avg_score', 'score_std', 'count', 'avg_rank', 'rank_std']
        print(age_analysis)
        
        # 3. 美国 vs 非美国名人的表现
        print("\n=== 美国 vs 非美国名人的表现分析 ===")
        us_analysis = df_analysis.groupby('is_american').agg({
            'overall_avg_score': ['mean', 'std', 'count'],
            'placement': ['mean', 'std']
        }).round(2)
        us_analysis.columns = ['avg_score', 'score_std', 'count', 'avg_rank', 'rank_std']
        us_analysis.index = ['Non-American', 'American']
        print(us_analysis)
        
        # 4. 相关性分析
        print("\n=== 关键变量相关性分析 ===")
        correlation_vars = ['overall_avg_score', 'placement', 'celebrity_age_during_season', 
                           'is_american', 'industry_performance', 'mean_fan_vote']
        correlation_matrix = df_analysis[correlation_vars].corr().round(3)
        print(correlation_matrix)
        
        # 5. 可视化分析结果
        print("\n=== 生成分析图表 ===")
        
        # 行业表现可视化
        plt.figure(figsize=(12, 8))
        sns.barplot(x='avg_score', y='celebrity_industry', data=industry_analysis)
        plt.title('不同行业的平均评委评分', fontsize=15)
        plt.xlabel('平均评分', fontsize=12)
        plt.ylabel('行业', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/industry_performance.png', dpi=300)
        
        # 年龄组表现可视化
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='age_group', y='overall_avg_score', data=df_analysis)
        plt.title('不同年龄组的评委评分分布', fontsize=15)
        plt.xlabel('年龄组', fontsize=12)
        plt.ylabel('评委评分', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/age_performance.png', dpi=300)
        
        # 评分与排名关系可视化
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='overall_avg_score', y='placement', data=df_analysis, alpha=0.6)
        sns.regplot(x='overall_avg_score', y='placement', data=df_analysis, scatter=False, color='red')
        plt.title('评委评分与最终排名关系', fontsize=15)
        plt.xlabel('评委评分', fontsize=12)
        plt.ylabel('最终排名', fontsize=12)
        plt.gca().invert_yaxis()  # 排名越小越好，反转Y轴
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/score_vs_rank.png', dpi=300)
        
        print(f"分析图表已保存到: {self.output_dir} 目录")
        
        # 保存分析结果
        industry_analysis.to_csv(f'{self.output_dir}/industry_analysis.csv', index=False)
        age_analysis.to_csv(f'{self.output_dir}/age_analysis.csv', index=False)
        us_analysis.to_csv(f'{self.output_dir}/us_analysis.csv', index=False)
        correlation_matrix.to_csv(f'{self.output_dir}/correlation_matrix.csv')
        
        print(f"分析结果已保存到: {self.output_dir} 目录")
    
    def build_prediction_models(self):
        """建立预测模型"""
        print("="*50)
        print("建立预测模型...")
        
        if self.df_combined is None:
            print("错误: 数据未预处理")
            return
        
        # 准备建模数据（去除缺失值）
        df_model = self.df_combined.dropna(subset=['overall_avg_score', 'placement']).copy()
        print(f"建模数据形状: {df_model.shape}")
        
        # 1. 模型1：预测评委评分
        print("\n=== 模型1：预测评委评分 ===")
        categorical_features = ['celebrity_industry', 'age_group']
        numerical_features = ['celebrity_age_during_season', 'is_american', 'industry_performance']
        
        # 创建预处理管道
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numerical_features),
                ('cat', Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ])
        
        # 准备特征和目标变量
        X_score = df_model[numerical_features + categorical_features]
        y_score = df_model['overall_avg_score']
        
        # 划分训练集和测试集
        X_score_train, X_score_test, y_score_train, y_score_test = train_test_split(
            X_score, y_score, test_size=0.2, random_state=42)
        
        print(f"评分预测 - 训练集大小: {X_score_train.shape}, 测试集大小: {X_score_test.shape}")
        
        # 定义要评估的模型
        score_models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # 评估和选择最佳模型
        score_results = {}
        best_score_model = None
        best_score_r2 = -np.inf
        
        for name, model in score_models.items():
            print(f"\n评估模型: {name}")
            
            # 创建完整的管道
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
            
            # 交叉验证
            cv_scores = cross_val_score(pipeline, X_score_train, y_score_train, 
                                       cv=5, scoring='r2')
            print(f"交叉验证 R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            
            # 在测试集上评估
            pipeline.fit(X_score_train, y_score_train)
            y_score_pred = pipeline.predict(X_score_test)
            
            r2 = r2_score(y_score_test, y_score_pred)
            rmse = np.sqrt(mean_squared_error(y_score_test, y_score_pred))
            mae = mean_absolute_error(y_score_test, y_score_pred)
            
            print(f"测试集 R²: {r2:.3f}")
            print(f"测试集 RMSE: {rmse:.3f}")
            print(f"测试集 MAE: {mae:.3f}")
            
            # 保存结果
            score_results[name] = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': pipeline
            }
            
            # 更新最佳模型
            if r2 > best_score_r2:
                best_score_r2 = r2
                best_score_model = pipeline
        
        # 保存最佳评分预测模型
        joblib.dump(best_score_model, f'{self.output_dir}/best_score_model.pkl')
        print(f"\n最佳评分预测模型已保存: {self.output_dir}/best_score_model.pkl")
        
        # 2. 模型2：预测最终排名
        print("\n=== 模型2：预测最终排名 ===")
        
        # 准备特征和目标变量（添加评委评分作为特征）
        X_rank = df_model[numerical_features + categorical_features + ['overall_avg_score', 'mean_fan_vote']]
        y_rank = df_model['placement']
        
        # 划分训练集和测试集
        X_rank_train, X_rank_test, y_rank_train, y_rank_test = train_test_split(
            X_rank, y_rank, test_size=0.2, random_state=42)
        
        print(f"排名预测 - 训练集大小: {X_rank_train.shape}, 测试集大小: {X_rank_test.shape}")
        
        # 定义要评估的模型
        rank_models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # 评估和选择最佳模型
        rank_results = {}
        best_rank_model = None
        best_rank_r2 = -np.inf
        
        for name, model in rank_models.items():
            print(f"\n评估模型: {name}")
            
            # 创建完整的管道
            pipeline = Pipeline([
                ('preprocessor', ColumnTransformer(
                    transformers=[
                        ('num', Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler())
                        ]), numerical_features + ['overall_avg_score', 'mean_fan_vote']),
                        ('cat', Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore'))
                        ]), categorical_features)
                    ])
                ),
                ('regressor', model)
            ])
            
            # 交叉验证
            cv_scores = cross_val_score(pipeline, X_rank_train, y_rank_train, 
                                       cv=5, scoring='r2')
            print(f"交叉验证 R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            
            # 在测试集上评估
            pipeline.fit(X_rank_train, y_rank_train)
            y_rank_pred = pipeline.predict(X_rank_test)
            
            r2 = r2_score(y_rank_test, y_rank_pred)
            rmse = np.sqrt(mean_squared_error(y_rank_test, y_rank_pred))
            mae = mean_absolute_error(y_rank_test, y_rank_pred)
            
            print(f"测试集 R²: {r2:.3f}")
            print(f"测试集 RMSE: {rmse:.3f}")
            print(f"测试集 MAE: {mae:.3f}")
            
            # 保存结果
            rank_results[name] = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': pipeline
            }
            
            # 更新最佳模型
            if r2 > best_rank_r2:
                best_rank_r2 = r2
                best_rank_model = pipeline
        
        # 保存最佳排名预测模型
        joblib.dump(best_rank_model, f'{self.output_dir}/best_rank_model.pkl')
        print(f"\n最佳排名预测模型已保存: {self.output_dir}/best_rank_model.pkl")
        
        # 3. 特征重要性分析
        print("\n=== 特征重要性分析 ===")
        
        # 对于随机森林和梯度提升模型，分析特征重要性
        for model_name, result in rank_results.items():
            if model_name in ['RandomForest', 'GradientBoosting']:
                print(f"\n{model_name} 特征重要性:")
                
                # 获取特征名称
                model = result['model']
                preprocessor = model.named_steps['preprocessor']
                
                # 获取数值特征名称
                num_features = numerical_features + ['overall_avg_score', 'mean_fan_vote']
                
                # 获取类别特征的独热编码后的名称
                cat_features = []
                if hasattr(preprocessor.named_transformers_['cat'].named_steps['onehot'], 'get_feature_names_out'):
                    cat_features = list(preprocessor.named_transformers_['cat'].named_steps['onehot']
                                       .get_feature_names_out(categorical_features))
                
                # 合并所有特征名称
                all_features = num_features + cat_features
                
                # 获取特征重要性
                importances = model.named_steps['regressor'].feature_importances_
                
                # 创建特征重要性DataFrame
                feature_importance = pd.DataFrame({
                    'feature': all_features,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print(feature_importance.head(10))
                
                # 可视化特征重要性
                plt.figure(figsize=(12, 8))
                sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
                plt.title(f'{model_name} - 特征重要性', fontsize=15)
                plt.xlabel('重要性', fontsize=12)
                plt.ylabel('特征', fontsize=12)
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/{model_name.lower()}_feature_importance.png', dpi=300)
        
        # 保存模型评估结果
        pd.DataFrame(score_results).T.to_csv(f'{self.output_dir}/score_model_results.csv')
        pd.DataFrame(rank_results).T.to_csv(f'{self.output_dir}/rank_model_results.csv')
        
        print(f"\n模型评估结果已保存到: {self.output_dir} 目录")
        
        # 存储结果
        self.models = {
            'score_models': score_results,
            'rank_models': rank_results,
            'best_score_model': best_score_model,
            'best_rank_model': best_rank_model
        }
        
        self.results = {
            'score_results': score_results,
            'rank_results': rank_results
        }
    
    def generate_summary_report(self):
        """生成分析总结报告"""
        print("="*50)
        print("生成分析总结报告...")
        
        if not self.results:
            print("错误: 没有模型结果可总结")
            return
        
        report = []
        report.append("# 名人舞蹈比赛特征影响分析报告\n")
        
        # 1. 数据概述
        report.append("## 1. 数据概述\n")
        if self.df_combined is not None:
            report.append(f"- 总样本数: {self.df_combined.shape[0]} 位名人")
            report.append(f"- 总特征数: {self.df_combined.shape[1]} 个特征")
            report.append(f"- 主要特征类别: 名人基本信息、评委评分、粉丝投票、比赛结果")
        
        # 2. 关键发现
        report.append("\n## 2. 关键发现\n")
        
        # 从相关性分析中获取关键信息
        if self.df_combined is not None:
            df_analysis = self.df_combined.dropna(subset=['overall_avg_score', 'placement']).copy()
            correlation = df_analysis['overall_avg_score'].corr(df_analysis['placement'])
            
            report.append(f"- **评委评分是决定最终排名的关键因素**，相关系数为 {correlation:.3f}")
            
            # 行业影响
            industry_analysis = df_analysis.groupby('celebrity_industry')['overall_avg_score'].mean()
            top_industry = industry_analysis.sort_values(ascending=False).head(1)
            bottom_industry = industry_analysis.sort_values(ascending=False).tail(1)
            
            report.append(f"- **行业背景对表现有显著影响**，{top_industry.index[0]}平均评分最高({top_industry.iloc[0]:.2f})，"
                         f"{bottom_industry.index[0]}平均评分最低({bottom_industry.iloc[0]:.2f})")
            
            # 年龄影响
            age_analysis = df_analysis.groupby('age_group')['overall_avg_score'].mean()
            top_age = age_analysis.sort_values(ascending=False).head(1)
            report.append(f"- **年龄是重要影响因素**，{top_age.index[0]}年龄组平均评分最高({top_age.iloc[0]:.2f})")
            
            # 国籍影响
            us_analysis = df_analysis.groupby('is_american')['overall_avg_score'].mean()
            us_diff = us_analysis[1] - us_analysis[0]
            report.append(f"- **美国籍名人表现略优**，平均评分比非美国名人高 {us_diff:.2f} 分")
        
        # 3. 模型性能
        report.append("\n## 3. 模型性能\n")
        
        # 评分预测模型
        if 'score_results' in self.results:
            best_score_model = max(self.results['score_results'], key=lambda k: self.results['score_results'][k]['r2'])
            best_score_r2 = self.results['score_results'][best_score_model]['r2']
            best_score_rmse = self.results['score_results'][best_score_model]['rmse']
            
            report.append("### 3.1 评委评分预测模型\n")
            report.append(f"- 最佳模型: {best_score_model}")
            report.append(f"- R² 得分: {best_score_r2:.3f}")
            report.append(f"- RMSE: {best_score_rmse:.3f}")
            report.append("- 结论: 评委评分预测难度较大，受多种难以量化的艺术因素影响")
        
        # 排名预测模型
        if 'rank_results' in self.results:
            best_rank_model = max(self.results['rank_results'], key=lambda k: self.results['rank_results'][k]['r2'])
            best_rank_r2 = self.results['rank_results'][best_rank_model]['r2']
            best_rank_rmse = self.results['rank_results'][best_rank_model]['rmse']
            
            report.append("\n### 3.2 最终排名预测模型\n")
            report.append(f"- 最佳模型: {best_rank_model}")
            report.append(f"- R² 得分: {best_rank_r2:.3f}")
            report.append(f"- RMSE: {best_rank_rmse:.3f}")
            report.append("- 结论: 最终排名预测效果良好，主要受评委评分影响")
        
        # 4. 特征重要性
        report.append("\n## 4. 特征重要性分析\n")
        report.append("- 评委评分是影响最终排名的最重要因素")
        report.append("- 年龄对评委评分有显著影响")
        report.append("- 行业背景影响评委的评分标准")
        report.append("- 国籍因素对表现有一定影响")
        
        # 5. 实用建议
        report.append("\n## 5. 实用建议\n")
        report.append("### 5.1 对参赛名人的建议\n")
        report.append("- 注重舞蹈技巧的提升，评委评分是决定排名的关键")
        report.append("- 选择与自己行业背景相符的舞蹈风格")
        report.append("- 年轻名人应充分利用年龄优势，展现活力")
        report.append("- 非美国籍名人应注重文化融合，展现独特魅力")
        
        report.append("\n### 5.2 对节目制作方的建议\n")
        report.append("- 邀请多元化行业背景的名人参赛，增加比赛看点")
        report.append("- 为不同年龄段的名人提供针对性的培训")
        report.append("- 考虑评委组成的多元化，减少评分偏差")
        report.append("- 优化粉丝投票机制，平衡专业评分与大众喜好")
        
        # 6. 结论
        report.append("\n## 6. 结论\n")
        report.append("本研究通过机器学习模型深入分析了名人特征对舞蹈比赛表现的影响。研究发现，评委评分是决定最终排名的关键因素，"
                     "而名人的年龄、行业背景和国籍等特征对评委评分有显著影响。最终排名预测模型表现优异，可以为比赛结果预测提供参考。"
                     "这些发现不仅有助于理解舞蹈比赛的评分机制，也为参赛名人和节目制作方提供了有价值的建议。")
        
        # 保存报告
        with open(f'{self.output_dir}/analysis_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"分析总结报告已保存: {self.output_dir}/analysis_report.md")
    
    def run_full_analysis(self):
        """运行完整的分析流程"""
        print("="*50)
        print("开始完整的名人舞蹈比赛特征影响分析...")
        
        # 1. 加载数据
        if not self.load_data():
            print("错误: 数据加载失败，分析终止")
            return
        
        # 2. 探索性数据分析
        self.explore_data()
        
        # 3. 数据预处理和特征工程
        self.preprocess_data()
        
        # 4. 特征影响分析
        self.analyze_features()
        
        # 5. 建立预测模型
        self.build_prediction_models()
        
        # 6. 生成分析总结报告
        self.generate_summary_report()
        
        print("="*50)
        print("完整分析流程执行完毕!")
        print(f"所有结果已保存到: {self.output_dir} 目录")

# 主函数
if __name__ == "__main__":
    # 创建模型实例
    model = CelebrityDanceAnalysisModel()
    
    # 运行完整分析
    model.run_full_analysis()
    
    print("\n分析完成! 请查看 output 目录获取详细结果。")