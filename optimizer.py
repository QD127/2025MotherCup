import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import os
import random
import math
import copy

class CourtYardOptimizer:
    def __init__(self, params: Dict[str, Any]):
        # 基础参数
        self.BASE_BUDGET = params.get('BASE_BUDGET', 20000000)
        self.COMMUNICATION_COST_PER_HOUSEHOLD = params.get('COMMUNICATION_COST', 30000)
        self.MAX_RENOVATION_COST_PER_HOUSEHOLD = params.get('MAX_RENOVATION_COST', 200000)
        self.MOVE_DURATION_MONTHS = params.get('MOVE_DURATION_MONTHS', 4)
        
        # 收入参数
        self.RENT_EW = params.get('RENT_EW', 8)
        self.RENT_NS = params.get('RENT_NS', 15)
        self.RENT_FULL_COURTYARD = params.get('RENT_FULL_COURTYARD', 30)
        self.ADJACENCY_BONUS_MULTIPLIER = params.get('ADJACENCY_BONUS', 1.2)
        
        # 其他常量
        self.MAX_AREA_INCREASE_RATIO = 1.3
        self.DAYS_PER_MONTH = 30
        self.MOVE_DURATION_DAYS = self.MOVE_DURATION_MONTHS * self.DAYS_PER_MONTH
        self.MAX_BUDGET = self.BASE_BUDGET * (1 + params.get('RESERVE_RATIO', 0.3))
        
        # 采光舒适度映射
        self.ORIENTATION_COMFORT = {'南': 4, '北': 4, '东': 3, '西': 2}
        
        # 模拟退火参数
        self.INITIAL_TEMPERATURE = 10000.0
        self.COOLING_RATE = 0.99
        self.MIN_TEMPERATURE = 1e-3
        self.MAX_ITERATIONS_PER_TEMP = 50
        
        # 数据存储
        self.df_info = None
        self.adjacency_dict = None
        self.potential_moves = None
        self.potential_moves_costs = None
        
        # 回调函数
        self.progress_callback = None
        self.log_callback = None

    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback
        
    def set_log_callback(self, callback):
        """设置日志回调函数"""
        self.log_callback = callback
        
    def _update_progress(self, value: float, message: str = None):
        """更新进度"""
        if self.progress_callback:
            self.progress_callback(value, message)
            
    def _log(self, message: str):
        """输出日志"""
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def load_data(self, info_path: str, adjacency_path: str) -> Tuple[pd.DataFrame, dict]:
        """加载数据文件"""
        try:
            self._log("正在读取地块信息表...")
            self.df_info = pd.read_excel(info_path)
            self._log(f"成功读取 {len(self.df_info)} 条地块记录")
            
            self._log("正在读取院落邻接表...")
            df_adjacency = pd.read_excel(adjacency_path)
            self._log(f"成功读取 {len(df_adjacency)} 条邻接关系")
            
            # 处理邻接关系
            self._log("正在处理院落邻接关系...")
            self.adjacency_dict = {}
            for _, row in df_adjacency.iterrows():
                courtyard_id = row['院落ID']
                neighbors_str = str(row['毗邻院落号']).strip()
                if neighbors_str and neighbors_str.lower() != 'nan':
                    try:
                        if neighbors_str.startswith('[') and neighbors_str.endswith(']'):
                            neighbors = [int(n.strip()) for n in neighbors_str[1:-1].split(',') if n.strip()]
                        else:
                            neighbors = [int(neighbors_str)]
                        
                        if courtyard_id not in self.adjacency_dict:
                            self.adjacency_dict[courtyard_id] = set()
                        self.adjacency_dict[courtyard_id].update(neighbors)
                        
                        for neighbor in neighbors:
                            if neighbor not in self.adjacency_dict:
                                self.adjacency_dict[neighbor] = set()
                            self.adjacency_dict[neighbor].add(courtyard_id)
                    except ValueError:
                        self._log(f"警告：无法解析院落 {courtyard_id} 的邻接信息")
            
            self._log(f"完成邻接关系处理，共 {len(self.adjacency_dict)} 个院落")
            
            # 预处理数据
            self._log("正在预处理数据...")
            self._preprocess_data()
            self._log("数据预处理完成")
            
            return self.df_info, self.adjacency_dict
            
        except Exception as e:
            self._log(f"数据加载错误: {str(e)}")
            return None, None

    def _preprocess_data(self):
        """预处理数据，计算潜在搬迁方案"""
        if self.df_info is None:
            return
            
        # 添加租金率和舒适度
        self._log("正在计算租金率和舒适度...")
        self.df_info['日租金率'] = self.df_info['地块方位'].apply(self._get_rent_rate)
        self.df_info['舒适度'] = self.df_info['地块方位'].apply(lambda x: self.ORIENTATION_COMFORT.get(x, 0))
        
        # 查找潜在搬迁方案
        self._log("正在分析潜在搬迁方案...")
        self.potential_moves = {}
        self.potential_moves_costs = {}
        
        residents = self.df_info[self.df_info['是否有住户'] == 1]
        empty_plots = self.df_info[self.df_info['是否有住户'] == 0]
        
        total_residents = len(residents)
        for i, (_, resident) in enumerate(residents.iterrows(), 1):
            self._update_progress(i / total_residents * 0.2, f"分析第 {i}/{total_residents} 户居民的搬迁方案...")
            
            possible_targets = []
            resident_id = resident['地块ID']
            resident_area = resident['地块面积']
            resident_comfort = resident['舒适度']
            resident_courtyard = resident['院落ID']
            resident_rent_rate = resident['日租金率']
            
            for _, target in empty_plots.iterrows():
                target_id = target['地块ID']
                target_area = target['地块面积']
                target_comfort = target['舒适度']
                target_courtyard = target['院落ID']
                target_rent_rate = target['日租金率']
                
                # 检查条件
                area_ok = (target_area >= resident_area) and (target_area <= self.MAX_AREA_INCREASE_RATIO * resident_area)
                comfort_ok = (target_comfort >= resident_comfort)
                courtyard_ok = (target_courtyard != resident_courtyard)
                
                if area_ok and comfort_ok and courtyard_ok:
                    # 计算成本
                    area_loss_cost = max(0, target_area - resident_area) * resident_rent_rate * self.MOVE_DURATION_DAYS
                    time_loss_cost = (resident_area * resident_rent_rate + target_area * target_rent_rate) * self.MOVE_DURATION_DAYS
                    
                    cost_details = (area_loss_cost, time_loss_cost)
                    possible_targets.append((target_id, cost_details))
            
            if possible_targets:
                self.potential_moves[resident_id] = possible_targets
                self.potential_moves_costs[resident_id] = {tid: costs for tid, costs in possible_targets}
                self._log(f"居民 {resident_id} 有 {len(possible_targets)} 个可选搬迁目标")
            else:
                self._log(f"警告：居民 {resident_id} 没有找到合适的搬迁目标")

    def _get_rent_rate(self, orientation: str) -> float:
        """获取租金率"""
        if orientation in ['南', '北']:
            return self.RENT_NS
        elif orientation in ['东', '西']:
            return self.RENT_EW
        return 0

    def _calculate_state_metrics(self, current_moves: Dict[int, int]) -> Dict[str, Any]:
        """计算给定搬迁方案的各项指标"""
        if not self.df_info is not None:
            return None
            
        total_cost = 0
        num_moved = len(current_moves)
        moved_resident_ids = list(current_moves.keys())
        target_plot_ids = list(current_moves.values())
        
        # 计算总成本
        for resident_id, target_id in current_moves.items():
            area_loss, time_loss = self.potential_moves_costs[resident_id][target_id]
            move_cost = (self.COMMUNICATION_COST_PER_HOUSEHOLD + 
                        self.MAX_RENOVATION_COST_PER_HOUSEHOLD + 
                        area_loss + time_loss)
            total_cost += move_cost
            
        budget_ok = (total_cost <= self.MAX_BUDGET)
        
        # 计算空置院落
        df_final_state = self.df_info.copy()
        df_final_state.loc[df_final_state['地块ID'].isin(moved_resident_ids), '是否有住户'] = 0
        df_final_state.loc[df_final_state['地块ID'].isin(target_plot_ids), '是否有住户'] = 1
        
        courtyard_occupancy = df_final_state.groupby('院落ID')['是否有住户'].sum()
        empty_courtyard_ids = courtyard_occupancy[courtyard_occupancy == 0].index.tolist()
        num_empty_courtyards = len(empty_courtyard_ids)
        
        # 计算空置院落总面积
        courtyard_areas = self.df_info.drop_duplicates(subset=['院落ID']).set_index('院落ID')['院落面积']
        total_empty_area = courtyard_areas.reindex(empty_courtyard_ids).sum()
        
        # 计算毗邻空置院落对数量
        adjacent_empty_pairs = 0
        checked_pairs = set()
        for cid1 in empty_courtyard_ids:
            if cid1 in self.adjacency_dict:
                for cid2 in self.adjacency_dict[cid1]:
                    if cid2 in empty_courtyard_ids and cid2 > cid1:
                        pair = tuple(sorted((cid1, cid2)))
                        if pair not in checked_pairs:
                            adjacent_empty_pairs += 1
                            checked_pairs.add(pair)
        
        # 计算最终日收入
        final_daily_income = self._calculate_daily_income(df_final_state, empty_courtyard_ids)
        
        # 计算能量值
        W_EMPTY_COURTYARDS = 1000000
        W_ADJACENT_PAIRS = 100000
        W_AREA = 1
        W_MOVES = 100
        W_BUDGET = 1
        
        budget_penalty = 0
        if not budget_ok:
            if total_cost > self.BASE_BUDGET:
                budget_penalty = (total_cost - self.BASE_BUDGET) * 100
            if total_cost > self.MAX_BUDGET:
                budget_penalty += (total_cost - self.MAX_BUDGET) * 1000
                
        energy = (
            - num_empty_courtyards * W_EMPTY_COURTYARDS
            - adjacent_empty_pairs * W_ADJACENT_PAIRS
            - total_empty_area * W_AREA
            + num_moved * W_MOVES
            + budget_penalty * W_BUDGET
        )
        
        return {
            'total_cost': total_cost,
            'num_moved': num_moved,
            'empty_courtyards': sorted(empty_courtyard_ids),
            'num_empty_courtyards': num_empty_courtyards,
            'adjacent_empty_pairs': adjacent_empty_pairs,
            'total_empty_area': total_empty_area,
            'final_daily_income': final_daily_income,
            'budget_ok': budget_ok,
            'energy': energy,
            'moves_list': current_moves
        }

    def _calculate_daily_income(self, df_state: pd.DataFrame, empty_courtyard_ids: List[int]) -> float:
        """计算给定状态下的每日收入"""
        final_daily_income = 0
        empty_courtyard_set = set(empty_courtyard_ids)
        processed_adjacent_courtyards = set()
        
        # 计算空置院落收入
        courtyard_areas = self.df_info.drop_duplicates(subset=['院落ID']).set_index('院落ID')['院落面积']
        
        for cid1 in empty_courtyard_ids:
            if cid1 in processed_adjacent_courtyards:
                continue
                
            is_adjacent_pair_member = False
            if cid1 in self.adjacency_dict:
                for cid2 in self.adjacency_dict[cid1]:
                    if cid2 in empty_courtyard_set and cid2 > cid1:
                        area1 = courtyard_areas.get(cid1, 0)
                        area2 = courtyard_areas.get(cid2, 0)
                        pair_income = (area1 + area2) * self.RENT_FULL_COURTYARD * self.ADJACENCY_BONUS_MULTIPLIER
                        final_daily_income += pair_income
                        processed_adjacent_courtyards.add(cid1)
                        processed_adjacent_courtyards.add(cid2)
                        is_adjacent_pair_member = True
                        break
                        
            if not is_adjacent_pair_member:
                area = courtyard_areas.get(cid1, 0)
                final_daily_income += area * self.RENT_FULL_COURTYARD
                processed_adjacent_courtyards.add(cid1)
        
        # 计算零散地块收入
        scattered_plots = df_state[~df_state['院落ID'].isin(empty_courtyard_set)]
        for _, plot in scattered_plots.iterrows():
            if plot['是否有住户'] == 0:
                final_daily_income += plot['地块面积'] * plot['日租金率']
                
        return final_daily_income

    def _get_neighbor_state(self, current_moves: Dict[int, int]) -> Dict[int, int]:
        """生成邻近状态"""
        neighbor_moves = copy.deepcopy(current_moves)
        occupied_targets = set(neighbor_moves.values())
        moved_residents = set(neighbor_moves.keys())
        all_potential_residents = set(self.potential_moves.keys())
        
        possible_actions = []
        if len(moved_residents) < len(all_potential_residents):
            possible_actions.append('add')
        if len(moved_residents) > 0:
            possible_actions.append('remove')
            possible_actions.append('change')
            
        if not possible_actions:
            return neighbor_moves
            
        action = random.choice(possible_actions)
        
        if action == 'add':
            unmoved_residents = list(all_potential_residents - moved_residents)
            if unmoved_residents:
                resident_to_add = random.choice(unmoved_residents)
                possible_targets = self.potential_moves[resident_to_add]
                available_targets = [(tid, cost) for tid, cost in possible_targets if tid not in occupied_targets]
                if available_targets:
                    target_id, _ = random.choice(available_targets)
                    neighbor_moves[resident_to_add] = target_id
                    
        elif action == 'remove':
            resident_to_remove = random.choice(list(moved_residents))
            del neighbor_moves[resident_to_remove]
            
        elif action == 'change':
            resident_to_change = random.choice(list(moved_residents))
            current_target = neighbor_moves[resident_to_change]
            possible_targets = self.potential_moves[resident_to_change]
            available_new_targets = [
                (tid, cost) for tid, cost in possible_targets
                if tid != current_target and tid not in occupied_targets
            ]
            if available_new_targets:
                new_target_id, _ = random.choice(available_new_targets)
                neighbor_moves[resident_to_change] = new_target_id
                
        return neighbor_moves

    def run_optimization(self) -> Dict[str, Any]:
        """运行优化算法"""
        if self.df_info is None or self.potential_moves is None:
            return {'status': 'error', 'message': '数据未正确加载'}
            
        self._log("开始运行模拟退火优化算法...")
        
        # 初始状态
        current_state_moves = {}
        current_metrics = self._calculate_state_metrics(current_state_moves)
        current_energy = current_metrics['energy']
        
        best_state_metrics = current_metrics
        temperature = self.INITIAL_TEMPERATURE
        
        history = []
        iteration = 0
        total_iterations = int(math.log(self.MIN_TEMPERATURE / self.INITIAL_TEMPERATURE, self.COOLING_RATE) * self.MAX_ITERATIONS_PER_TEMP)
        
        self._log(f"预计总迭代次数: {total_iterations}")
        
        while temperature > self.MIN_TEMPERATURE:
            accepted_in_temp = 0
            for _ in range(self.MAX_ITERATIONS_PER_TEMP):
                iteration += 1
                progress = iteration / total_iterations
                self._update_progress(0.2 + progress * 0.8, 
                                   f"优化进度: {progress*100:.1f}% (温度: {temperature:.2f})")
                
                neighbor_moves = self._get_neighbor_state(current_state_moves)
                neighbor_metrics = self._calculate_state_metrics(neighbor_moves)
                neighbor_energy = neighbor_metrics['energy']
                
                delta_energy = neighbor_energy - current_energy
                
                accept = False
                if delta_energy < 0:
                    accept = True
                else:
                    acceptance_probability = math.exp(-delta_energy / temperature)
                    if random.random() < acceptance_probability:
                        accept = True
                        
                if accept:
                    accepted_in_temp += 1
                    current_state_moves = neighbor_moves
                    current_metrics = neighbor_metrics
                    current_energy = neighbor_energy
                    
                    update_best = False
                    if neighbor_metrics['budget_ok']:
                        if not best_state_metrics['budget_ok']:
                            update_best = True
                        else:
                            if neighbor_metrics['num_empty_courtyards'] > best_state_metrics['num_empty_courtyards']:
                                update_best = True
                            elif neighbor_metrics['num_empty_courtyards'] == best_state_metrics['num_empty_courtyards']:
                                if neighbor_metrics['adjacent_empty_pairs'] > best_state_metrics['adjacent_empty_pairs']:
                                    update_best = True
                                elif neighbor_metrics['adjacent_empty_pairs'] == best_state_metrics['adjacent_empty_pairs']:
                                    if neighbor_metrics['total_empty_area'] > best_state_metrics['total_empty_area']:
                                        update_best = True
                                    elif neighbor_metrics['total_empty_area'] == best_state_metrics['total_empty_area']:
                                        if neighbor_metrics['num_moved'] < best_state_metrics['num_moved']:
                                            update_best = True
                                        elif neighbor_metrics['num_moved'] == best_state_metrics['num_moved']:
                                            if neighbor_metrics['total_cost'] < best_state_metrics['total_cost']:
                                                update_best = True
                                                
                    if update_best:
                        best_state_metrics = copy.deepcopy(neighbor_metrics)
                        self._log(f"找到更优解：空置院落 {best_state_metrics['num_empty_courtyards']} 个，"
                                f"总成本 {best_state_metrics['total_cost']:,.0f} 元")
                        
                if iteration % 20 == 0:
                    history.append({
                        'iteration': iteration,
                        'temperature': temperature,
                        'energy': current_energy,
                        'num_empty_courtyards': current_metrics['num_empty_courtyards'],
                        'total_cost': current_metrics['total_cost']
                    })
                    
            temperature *= self.COOLING_RATE
            self._log(f"温度降至 {temperature:.2f}，当前阶段接受率: {accepted_in_temp/self.MAX_ITERATIONS_PER_TEMP:.1%}")
            
        # 添加最终状态到历史记录
        history.append({
            'iteration': iteration + 1,
            'temperature': temperature,
            'energy': best_state_metrics['energy'],
            'num_empty_courtyards': best_state_metrics['num_empty_courtyards'],
            'total_cost': best_state_metrics['total_cost']
        })
        
        self._log("优化完成！")
        best_state_metrics['status'] = 'success'
        best_state_metrics['history'] = history
        return best_state_metrics

    def generate_plots(self, results: Dict[str, Any], output_dir: str):
        """生成可视化图表"""
        if results['status'] != 'success':
            return
            
        history = results.get('history', [])
        if not history:
            return
            
        # 优化进度图
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        iterations = [item['iteration'] for item in history]
        energy = [item['energy'] for item in history]
        num_empty = [item['num_empty_courtyards'] for item in history]
        costs = [item['total_cost'] for item in history]
        
        axs[0].plot(iterations, energy, label='系统能量', color='blue')
        axs[0].set_ylabel('能量值')
        axs[0].set_title('模拟退火优化过程')
        axs[0].grid(True)
        axs[0].legend()
        
        axs[1].plot(iterations, num_empty, label='完整空置院落数量', color='green')
        axs[1].set_ylabel('数量')
        axs[1].grid(True)
        axs[1].legend()
        
        axs[2].plot(iterations, costs, label='总成本', color='red')
        axs[2].axhline(y=self.BASE_BUDGET, color='orange', linestyle='--', 
                       label=f'保底预算 ({self.BASE_BUDGET:,.0f})')
        axs[2].axhline(y=self.MAX_BUDGET, color='purple', linestyle=':', 
                       label=f'最高预算 ({self.MAX_BUDGET:,.0f})')
        axs[2].set_xlabel('迭代次数')
        axs[2].set_ylabel('成本 (元)')
        axs[2].grid(True)
        axs[2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optimization_progress.png'))
        plt.close()
        
        # 财务总结图
        initial_metrics = self._calculate_state_metrics({})
        initial_income = initial_metrics['final_daily_income']
        final_income = results['final_daily_income']
        total_cost = results['total_cost']
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        income_labels = ['初始日收入', '最终日收入']
        income_values = [initial_income, final_income]
        bars_income = axs[0].bar(income_labels, income_values, color=['gray', 'green'])
        axs[0].set_ylabel('收入 (元/天)')
        axs[0].set_title('每日收入对比')
        axs[0].bar_label(bars_income, fmt='{:,.2f}')
        
        cost_labels = ['总成本', '保底预算', '最高预算']
        cost_values = [total_cost, self.BASE_BUDGET, self.MAX_BUDGET]
        colors_cost = ['red' if total_cost > self.MAX_BUDGET else 
                      ('orange' if total_cost > self.BASE_BUDGET else 'blue'),
                      'orange', 'purple']
        bars_cost = axs[1].bar(cost_labels, cost_values, color=colors_cost)
        axs[1].set_ylabel('金额 (元)')
        axs[1].set_title('总成本与预算')
        axs[1].bar_label(bars_cost, fmt='{:,.0f}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'financial_summary.png'))
        plt.close()

    def format_results(self, results: Dict[str, Any]) -> str:
        """格式化结果文本"""
        if results['status'] != 'success':
            return "优化失败，请检查输入参数和数据。"
            
        text = "优化结果摘要：\n\n"
        
        # 搬迁方案
        text += f"共搬迁 {results['num_moved']} 户居民\n"
        if results['moves_list']:
            text += "\n搬迁方案:\n"
            for resident_id, target_id in results['moves_list'].items():
                res_info = self.df_info[self.df_info['地块ID'] == resident_id].iloc[0]
                tar_info = self.df_info[self.df_info['地块ID'] == target_id].iloc[0]
                text += f"院落 {res_info['院落ID']} 的居民 (地块 {resident_id}) -> "
                text += f"院落 {tar_info['院落ID']} 的空地块 {target_id}\n"
                
        # 空置院落
        text += f"\n成功腾出 {results['num_empty_courtyards']} 个完整院落: "
        text += f"{results['empty_courtyards']}\n"
        text += f"相互毗邻的空置院落对数量: {results['adjacent_empty_pairs']}\n"
        text += f"腾出的完整院落总面积: {results['total_empty_area']:.2f} 平方米\n"
        
        # 成本分析
        text += f"\n总成本: {results['total_cost']:,.2f} 元\n"
        if results['total_cost'] <= self.BASE_BUDGET:
            text += "成本在保底规划成本内\n"
        elif results['total_cost'] <= self.MAX_BUDGET:
            text += "成本在最高预算内，动用了部分备用金\n"
        else:
            text += "警告：成本超出最高预算！\n"
            
        # 收益分析
        initial_metrics = self._calculate_state_metrics({})
        initial_income = initial_metrics['final_daily_income']
        final_income = results['final_daily_income']
        income_change = final_income - initial_income
        
        text += f"\n每日收入变化:\n"
        text += f"初始状态: {initial_income:,.2f} 元/天\n"
        text += f"最终状态: {final_income:,.2f} 元/天\n"
        text += f"日收入增加: {income_change:,.2f} 元/天\n"
        
        if income_change > 0 and results['total_cost'] > 0:
            payback_days = results['total_cost'] / income_change
            text += f"\n预计投资回收期: {payback_days:.1f} 天 (约 {payback_days/30:.1f} 个月)\n"
            
        return text 