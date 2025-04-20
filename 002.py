import pandas as pd
import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches # For drawing shapes

# --- Font Configuration (尝试配置中文字体) ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # Or ['Microsoft YaHei'] etc. based on your system
    plt.rcParams['axes.unicode_minus'] = False
    print("中文 SimHei Font configuration attempted.")
except Exception as e:
    print(f"警告：未能设置中文字体: {e}。标签可能无法正确显示。")
    print(f"Warning: Failed to set Chinese font: {e}. Labels might not display correctly.")


# --- [PREVIOUS CODE: Constants, load_data, Helper Functions (get_rent_rate, etc.), find_potential_moves, calculate_state_metrics, get_neighbor_state] ---
# --- [在此处粘贴之前脚本的所有代码，从 配置参数 到 get_neighbor_state 函数结束] ---
# --- [PASTE ALL PREVIOUS CODE HERE: From Configuration Parameters up to the end of the get_neighbor_state function] ---
# --- 配置参数 (Configuration Parameters) ---

# 文件路径 (File Paths)
FILE_PATH_INFO = '附件1.xlsx' # 地块信息表 (Plot Information Table)
FILE_PATH_ADJACENCY = '附件2.xlsx' # 院落邻接表 (Courtyard Adjacency Table)

# 成本参数 (Cost Parameters)
COMMUNICATION_COST_PER_HOUSEHOLD = 30000  # 每户沟通成本 (Communication cost per household)
MAX_RENOVATION_COST_PER_HOUSEHOLD = 200000 # 每户最高修缮成本 (Max renovation cost per household)
# 注意：面积损失和时间损失成本将在计算中动态确定 (Note: Area loss and time loss costs are dynamically calculated)

# 收入参数 (Income Parameters)
RENT_EW = 8  # 东西厢房日租金 (Daily rent for East/West facing plots) 元/平米/天
RENT_NS = 15 # 南北厢房日租金 (Daily rent for North/South facing plots) 元/平米/天
RENT_FULL_COURTYARD = 30 # 完整空置院落日租金 (Daily rent for a complete empty courtyard) 元/平米/天
ADJACENCY_BONUS_MULTIPLIER = 1.2 # 毗邻院落租金加成系数 (Rent bonus multiplier for adjacent empty courtyards)

# 搬迁约束 (Relocation Constraints)
MAX_AREA_INCREASE_RATIO = 1.3 # 面积补偿上限比例 (Max area increase ratio for compensation)
MOVE_DURATION_MONTHS = 4 # 搬迁耗时（月）(Duration of relocation in months)
DAYS_PER_MONTH = 30 # 用于时间损失计算的每月天数 (Days per month for time loss calculation)
MOVE_DURATION_DAYS = MOVE_DURATION_MONTHS * DAYS_PER_MONTH # 搬迁耗时（天）(Duration of relocation in days)

# 预算限制 (Budget Constraints)
BASE_BUDGET = 20000000 # 保底规划成本 (Base budget)
MAX_BUDGET = BASE_BUDGET * 1.3 # 最高预算（含备用金）(Maximum budget including reserve)

# 采光舒适度映射 (Orientation Comfort Mapping)
# 正南=3, 正北=3, 东厢=2, 西厢=1 (South=3, North=3, East=2, West=1)
ORIENTATION_COMFORT = {'南': 4, '北': 4, '东': 3, '西': 2}

# 模拟退火参数 (Simulated Annealing Parameters)
INITIAL_TEMPERATURE = 10000.0 # 初始温度
COOLING_RATE = 0.99 # 冷却率
MIN_TEMPERATURE = 1e-3 # 最小温度
MAX_ITERATIONS_PER_TEMP = 50 # 每个温度下的迭代次数

# --- 数据加载与预处理 (Data Loading and Preprocessing) ---

def load_data(info_path, adjacency_path):
    """
    加载地块信息和院落邻接关系数据。
    Loads plot information and courtyard adjacency data.
    """
    try:
        df_info = pd.read_excel(info_path)
        df_adjacency = pd.read_excel(adjacency_path)
        print("成功加载数据文件。 (Successfully loaded data files.)")
    except FileNotFoundError:
        print(f"错误：找不到文件 {info_path} 或 {adjacency_path}。请确保文件在正确的路径下。")
        print(f"Error: File not found {info_path} or {adjacency_path}. Please ensure files are in the correct path.")
        return None, None

    # 预处理邻接关系数据 (Preprocess adjacency data)
    adjacency_dict = {}
    for _, row in df_adjacency.iterrows():
        courtyard_id = row['院落ID']
        # 处理可能的字符串列表 '[2]' 或数字 2 (Handle potential string list '[2]' or number 2)
        neighbors_str = str(row['毗邻院落号']).strip()
        if neighbors_str and neighbors_str.lower() != 'nan':
            try:
                # 尝试解析列表格式 (Try parsing list format)
                if neighbors_str.startswith('[') and neighbors_str.endswith(']'):
                    neighbors = [int(n.strip()) for n in neighbors_str[1:-1].split(',') if n.strip()]
                else: # 假设是单个数字 (Assume single number)
                    neighbors = [int(neighbors_str)]

                # 建立双向邻接关系 (Establish bidirectional adjacency)
                if courtyard_id not in adjacency_dict:
                    adjacency_dict[courtyard_id] = set()
                adjacency_dict[courtyard_id].update(neighbors)

                for neighbor in neighbors:
                    if neighbor not in adjacency_dict:
                        adjacency_dict[neighbor] = Xset()
                    adjacency_dict[neighbor].add(courtyard_id)

            except ValueError:
                print(f"警告：无法解析院落 {courtyard_id} 的邻接信息 '{row['毗邻院落号']}'。将跳过此条目。")
                print(f"Warning: Could not parse adjacency info '{row['毗邻院落号']}' for courtyard {courtyard_id}. Skipping this entry.")

    # 确保所有在 df_info 中的院落ID都在 adjacency_dict 中，即使没有邻居 (Ensure all courtyard IDs from df_info are in adjacency_dict, even if they have no neighbors)
    all_courtyard_ids = df_info['院落ID'].unique()
    for cid in all_courtyard_ids:
        if cid not in adjacency_dict:
            adjacency_dict[cid] = set() # 添加没有邻居的院落 (Add courtyards with no neighbors)

    print(f"邻接关系处理完成。共处理 {len(adjacency_dict)} 个院落的邻接信息。")
    print(f"Adjacency processing complete. Processed adjacency info for {len(adjacency_dict)} courtyards.")

    return df_info, adjacency_dict

# --- 辅助函数 (Helper Functions) ---

def get_rent_rate(orientation):
    """根据地块方位获取日租金率 (Get daily rent rate based on plot orientation)"""
    if orientation in ['南', '北']:
        return RENT_NS
    elif orientation in ['东', '西']:
        return RENT_EW
    else:
        return 0 # 未知方位或其他情况 (Unknown orientation or other cases)

def get_orientation_comfort(orientation):
    """获取方位的舒适度评分 (Get comfort score for orientation)"""
    return ORIENTATION_COMFORT.get(orientation, 0) # 未知方位舒适度为0 (Unknown orientation gets comfort 0)

def find_potential_moves(df_info):
    """
    为每个住户找到所有符合条件的潜在迁入空地块。
    Find all potential empty plots for each resident that meet the criteria.

    Returns:
        dict: {resident_plot_id: [(target_plot_id, cost_details), ...]}
              cost_details: (area_loss_cost, time_loss_cost)
    """
    residents = df_info[df_info['是否有住户'] == 1].copy()
    empty_plots = df_info[df_info['是否有住户'] == 0].copy()
    potential_moves = {}

    # 为地块信息添加租金率和舒适度，方便计算 (Add rent rate and comfort score to plot info for easier calculation)
    df_info['日租金率'] = df_info['地块方位'].apply(get_rent_rate)
    df_info['舒适度'] = df_info['地块方位'].apply(get_orientation_comfort)

    residents['日租金率'] = residents['地块方位'].apply(get_rent_rate)
    residents['舒适度'] = residents['地块方位'].apply(get_orientation_comfort)
    empty_plots['日租金率'] = empty_plots['地块方位'].apply(get_rent_rate)
    empty_plots['舒适度'] = empty_plots['地块方位'].apply(get_orientation_comfort)

    for _, resident in residents.iterrows():
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

            # 条件检查 (Check conditions)
            # 1. 面积补偿：新面积 >= 旧面积 AND 新面积 <= 1.3 * 旧面积
            area_ok = (target_area >= resident_area) and (target_area <= MAX_AREA_INCREASE_RATIO * resident_area)
            # 2. 采光补偿：新舒适度 >= 旧舒适度
            comfort_ok = (target_comfort >= resident_comfort)
            # 3. 不能搬到自己当前所在的院落（为了腾空院落）(Cannot move within the same courtyard - to empty it)
            courtyard_ok = (target_courtyard != resident_courtyard)

            if area_ok and comfort_ok and courtyard_ok:
                # 计算该潜在移动的成本组成部分 (Calculate cost components for this potential move)
                # 面积损失成本: (新面积 - 旧面积) * 旧地块日租金率 * 搬迁天数 (Area Loss Cost)
                area_loss_cost = max(0, target_area - resident_area) * resident_rent_rate * MOVE_DURATION_DAYS
                # 时间损失成本: (旧地块面积 * 旧地块日租金率 + 新地块面积 * 新地块日租金率) * 搬迁天数 (Time Loss Cost)
                time_loss_cost = (resident_area * resident_rent_rate + target_area * target_rent_rate) * MOVE_DURATION_DAYS

                cost_details = (area_loss_cost, time_loss_cost)
                possible_targets.append((target_id, cost_details))

        if possible_targets:
            potential_moves[resident_id] = possible_targets
            # print(f"住户地块 {resident_id} 找到 {len(possible_targets)} 个潜在迁入点。")
            # print(f"Resident plot {resident_id} found {len(possible_targets)} potential target plots.")

    print(f"共找到 {len(potential_moves)} 户居民的可行搬迁方案。")
    print(f"Found potential moves for {len(potential_moves)} residents.")
    return potential_moves, df_info # 返回更新了租金率和舒适度的df_info (Return updated df_info)

def calculate_state_metrics(current_moves, df_info, adjacency_dict, potential_moves_costs):
    """
    计算给定搬迁方案（状态）的各项指标：成本、空置院落、面积、邻接、收入等。
    Calculates metrics for a given relocation plan (state): cost, empty courtyards, area, adjacency, income, etc.

    Args:
        current_moves (dict): {resident_plot_id: target_plot_id} 当前搬迁方案
        df_info (pd.DataFrame): 包含所有地块信息（已添加租金率和舒适度）
        adjacency_dict (dict): 院落邻接关系
        potential_moves_costs (dict): {resident_plot_id: {target_plot_id: (area_loss, time_loss)}} 缓存的成本

    Returns:
        dict: 包含各项指标的字典 (Dictionary containing all metrics)
            'total_cost': 总成本
            'num_moved': 搬迁户数
            'empty_courtyards': 完全空置的院落ID列表
            'num_empty_courtyards': 完全空置院落数量
            'adjacent_empty_pairs': 毗邻空置院落对数量
            'total_empty_area': 完全空置院落总面积
            'final_daily_income': 搬迁后的总日收入
            'budget_ok': 是否在预算内 (True/False)
            'penalty': 用于优化的惩罚项 (Penalty term for optimization)
    """
    total_cost = 0
    num_moved = len(current_moves)
    moved_resident_ids = list(current_moves.keys())
    target_plot_ids = list(current_moves.values())

    # 1. 计算总成本 (Calculate Total Cost)
    for resident_id, target_id in current_moves.items():
        # 从缓存获取面积和时间损失成本 (Get area and time loss costs from cache)
        try:
             area_loss, time_loss = potential_moves_costs[resident_id][target_id]
        except KeyError:
             # 如果缓存缺失（理论上不应发生），重新计算 (Recalculate if cache miss - shouldn't happen in theory)
             print(f"警告：住户 {resident_id} 到 地块 {target_id} 的成本缓存缺失，重新计算。")
             print(f"Warning: Cost cache missing for resident {resident_id} to plot {target_id}, recalculating.")
             res_info = df_info[df_info['地块ID'] == resident_id].iloc[0]
             tar_info = df_info[df_info['地块ID'] == target_id].iloc[0]
             area_loss = max(0, tar_info['地块面积'] - res_info['地块面积']) * res_info['日租金率'] * MOVE_DURATION_DAYS
             time_loss = (res_info['地块面积'] * res_info['日租金率'] + tar_info['地块面积'] * tar_info['日租金率']) * MOVE_DURATION_DAYS

        # 每户固定成本 + 损失成本 (Fixed costs per household + loss costs)
        move_cost = COMMUNICATION_COST_PER_HOUSEHOLD + MAX_RENOVATION_COST_PER_HOUSEHOLD + area_loss + time_loss
        total_cost += move_cost

    budget_ok = (total_cost <= MAX_BUDGET)
    # 预算惩罚：如果超预算，则增加一个巨大的惩罚值，使其在优化中被强烈排斥
    # Budget penalty: Add a large penalty if over budget to strongly discourage it in optimization
    budget_penalty = 0
    if not budget_ok:
        # 优先保证基础预算，超出部分惩罚更高 (Prioritize base budget, penalize exceeding it more)
        if total_cost > BASE_BUDGET:
             budget_penalty = (total_cost - BASE_BUDGET) * 100 # 超出基础预算的惩罚 (Penalty for exceeding base budget)
        if total_cost > MAX_BUDGET:
             budget_penalty += (total_cost - MAX_BUDGET) * 1000 # 超出最大预算的惩罚更高 (Higher penalty for exceeding max budget)


    # 2. 确定搬迁后的地块占用情况 (Determine plot occupancy after moves)
    df_final_state = df_info.copy()
    # 标记搬出的地块为空置 (Mark vacated plots as empty)
    df_final_state.loc[df_final_state['地块ID'].isin(moved_resident_ids), '是否有住户'] = 0
    # 标记迁入的地块为占用 (Mark target plots as occupied)
    df_final_state.loc[df_final_state['地块ID'].isin(target_plot_ids), '是否有住户'] = 1

    # 3. 识别完全空置的院落 (Identify completely empty courtyards)
    courtyard_occupancy = df_final_state.groupby('院落ID')['是否有住户'].sum()
    empty_courtyard_ids = courtyard_occupancy[courtyard_occupancy == 0].index.tolist()
    num_empty_courtyards = len(empty_courtyard_ids)

    # 4. 计算空置院落总面积 (Calculate total area of empty courtyards)
    # 需要原始的院落面积信息，因为df_info只包含地块面积 (Need original courtyard area info)
    # 我们从原始df_info中获取每个院落的总面积 (Get total area for each courtyard from original df_info)
    # 注意：附件1预览显示院落面积在每行重复，我们取第一次出现的值 (Note: Preview shows courtyard area repeated per row, we take the first occurrence)
    courtyard_areas = df_info.drop_duplicates(subset=['院落ID']).set_index('院落ID')['院落面积']
    total_empty_area = courtyard_areas.reindex(empty_courtyard_ids).sum()

    # 5. 计算毗邻的空置院落对数量 (Calculate number of adjacent empty courtyard pairs)
    adjacent_empty_pairs = 0
    checked_pairs = set() # 防止重复计算 (Prevent double counting)
    for cid1 in empty_courtyard_ids:
        if cid1 in adjacency_dict:
            for cid2 in adjacency_dict[cid1]:
                if cid2 in empty_courtyard_ids and cid2 > cid1: # 确保cid2也在空院落列表且避免重复对(如1-2和2-1)
                     # Make sure cid2 is also empty and avoid duplicate pairs (like 1-2 and 2-1)
                    pair = tuple(sorted((cid1, cid2)))
                    if pair not in checked_pairs:
                         adjacent_empty_pairs += 1
                         checked_pairs.add(pair)

    # 6. 计算最终日收入 (Calculate final daily income)
    final_daily_income = 0
    empty_courtyard_set = set(empty_courtyard_ids)
    processed_adjacent_courtyards = set() # 跟踪已计算过加成的院落 (Track courtyards already processed for adjacency bonus)

    # 收入来自：完整空置院落（可能有邻接加成）+ 零散地块
    # Income from: Complete empty courtyards (possibly with adjacency bonus) + scattered plots

    # 计算空置院落收入 (Calculate income from empty courtyards)
    for cid1 in empty_courtyard_ids:
         if cid1 in processed_adjacent_courtyards:
             continue # 如果已作为邻接对的一部分计算过，跳过 (Skip if already calculated as part of an adjacent pair)

         is_adjacent_pair_member = False
         if cid1 in adjacency_dict:
             for cid2 in adjacency_dict[cid1]:
                 if cid2 in empty_courtyard_set and cid2 > cid1: # 发现邻接的空院落对 (Found an adjacent empty pair)
                     area1 = courtyard_areas.get(cid1, 0)
                     area2 = courtyard_areas.get(cid2, 0)
                     pair_income = (area1 + area2) * RENT_FULL_COURTYARD * ADJACENCY_BONUS_MULTIPLIER
                     final_daily_income += pair_income
                     processed_adjacent_courtyards.add(cid1)
                     processed_adjacent_courtyards.add(cid2)
                     is_adjacent_pair_member = True
                     break # 处理完这对就不再找cid1的其他邻居了 (Processed this pair, stop looking for other neighbors of cid1)

         # 如果不是邻接对的一部分，按普通完整院落计算 (If not part of an adjacent pair, calculate as a regular complete courtyard)
         if not is_adjacent_pair_member:
             area = courtyard_areas.get(cid1, 0)
             final_daily_income += area * RENT_FULL_COURTYARD
             processed_adjacent_courtyards.add(cid1) # 标记为已处理 (Mark as processed)


    # 计算零散地块收入 (Calculate income from scattered plots)
    # 零散地块 = 不在完整空置院落中的所有地块 (Scattered plots = all plots not in a complete empty courtyard)
    scattered_plots = df_final_state[~df_final_state['院落ID'].isin(empty_courtyard_set)]
    for _, plot in scattered_plots.iterrows():
        # 只有空置的零散地块才产生租金（题目似乎暗示如此，住人的不产生）
        # Only empty scattered plots generate rent (problem seems to imply this, occupied ones don't)
        # 修正：题目描述“直接出租分散的产权地块给开发商带来收入为...”，未区分是否住人，按地块本身性质算
        # Correction: Prompt says "directly renting out scattered plots brings income...", doesn't distinguish occupancy. Calculate based on plot type.
        # 再修正：搬迁后的状态，原住户地块如果是零散的且空了，应产生租金；新迁入的地块如果是零散的，现在住人了不产生租金。
        # Re-correction: In the final state, if an original resident plot is now scattered and empty, it should generate rent. If a target plot is now scattered and occupied, it does not.
        # 最清晰的理解：计算所有*当前状态下*空置地块的租金，但如果该地块属于已按“完整院落”计算过租金的院落，则跳过。
        # Clearest interpretation: Calculate rent for all plots that are *currently empty*, but skip if the plot belongs to a courtyard already accounted for as a "complete empty courtyard".

        if plot['是否有住户'] == 0: # 只计算空地块 (Only consider empty plots)
             # 检查地块所在院落是否已按完整空置计算 (Check if the plot's courtyard was calculated as fully empty)
             # 这一步在上面筛选 scattered_plots 时已经完成，这里可以直接加
             # This check was already done when filtering scattered_plots, can add directly here.
             final_daily_income += plot['地块面积'] * plot['日租金率']


    # 7. 计算综合惩罚项/能量值 (Calculate combined penalty/energy value for SA)
    # 目标：最大化空院落数 -> 邻接对数 -> 总面积，最小化搬迁数，满足预算
    # Objective: Maximize empty courtyards -> adjacent pairs -> total area, Minimize moves, satisfy budget
    # SA 通常最小化能量，所以我们将最大化目标取负，最小化目标保留正
    # SA usually minimizes energy, so we negate maximization goals and keep minimization goals positive.

    # 使用足够大的权重来确保优先级 (Use large enough weights to ensure priority)
    # 权重需要根据各指标的量级调整，这里是示例值 (Weights need adjustment based on metric scales, these are example values)
    W_EMPTY_COURTYARDS = 1000000
    W_ADJACENT_PAIRS = 100000
    W_AREA = 1 # 面积权重相对较小，在前两者之后考虑 (Area weight is relatively small, considered after the first two)
    W_MOVES = 100 # 惩罚搬迁数量 (Penalize number of moves)
    W_BUDGET = 1 # 预算惩罚已在 budget_penalty 中计算 (Budget penalty already calculated in budget_penalty)

    # 能量值 (Energy Value)
    energy = (
        - num_empty_courtyards * W_EMPTY_COURTYARDS
        - adjacent_empty_pairs * W_ADJACENT_PAIRS
        - total_empty_area * W_AREA
        + num_moved * W_MOVES
        + budget_penalty * W_BUDGET # 直接加预算惩罚 (Directly add budget penalty)
        # 可选：如果希望严格在基础预算内，可以对超出基础预算的部分也加一个惩罚
        # Optional: If strictly within base budget is desired, add penalty for exceeding base budget too
        # + max(0, total_cost - BASE_BUDGET) * SOME_WEIGHT
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
        'moves_list': current_moves # 方便后续使用 (Convenient for later use)
    }


def get_neighbor_state(current_moves, potential_moves, df_info):
    """
    生成一个邻近状态（搬迁方案）。
    Generates a neighboring state (relocation plan).

    策略 Strategy:
    1. 随机选择一个操作：添加搬迁、移除搬迁、更改搬迁目的地。
       Randomly choose an action: add a move, remove a move, change a move destination.
    2. 执行操作，确保不违反基本规则（如目标地块已被占用）。
       Execute the action, ensuring basic rules are not violated (e.g., target plot already occupied).
    """
    neighbor_moves = copy.deepcopy(current_moves)
    occupied_targets = set(neighbor_moves.values())
    moved_residents = set(neighbor_moves.keys())
    all_residents_with_options = list(potential_moves.keys())
    all_potential_residents = set(all_residents_with_options) # 可以搬迁的住户集合

    possible_actions = []
    if len(moved_residents) < len(all_potential_residents):
        possible_actions.append('add') # 如果还有未搬迁的住户可以搬，则可以添加
    if len(moved_residents) > 0:
        possible_actions.append('remove') # 如果有已搬迁的住户，可以移除
        possible_actions.append('change') # 如果有已搬迁的住户，可以尝试更改

    if not possible_actions:
        return neighbor_moves # 没有可执行的操作 (No possible actions)

    action = random.choice(possible_actions)

    if action == 'add':
        # 尝试添加一个未搬迁住户的搬迁
        unmoved_residents = list(all_potential_residents - moved_residents)
        if unmoved_residents:
            resident_to_add = random.choice(unmoved_residents)
            possible_targets = potential_moves[resident_to_add]
            # 过滤掉已被占用的目标地块 (Filter out already occupied target plots)
            available_targets = [(tid, cost) for tid, cost in possible_targets if tid not in occupied_targets]
            if available_targets:
                target_id, _ = random.choice(available_targets)
                neighbor_moves[resident_to_add] = target_id
                # print(f"动作：添加搬迁 {resident_to_add} -> {target_id}")

    elif action == 'remove':
        # 移除一个已有的搬迁
        resident_to_remove = random.choice(list(moved_residents))
        del neighbor_moves[resident_to_remove]
        # print(f"动作：移除搬迁 {resident_to_remove}")


    elif action == 'change':
        # 更改一个现有搬迁的目的地
        resident_to_change = random.choice(list(moved_residents))
        current_target = neighbor_moves[resident_to_change]
        possible_targets = potential_moves[resident_to_change]
        # 查找除了当前目标和已被占用的目标之外的其他可用目标
        # Find other available targets excluding the current one and those already occupied
        available_new_targets = [
            (tid, cost) for tid, cost in possible_targets
            if tid != current_target and tid not in occupied_targets
        ]
        if available_new_targets:
            new_target_id, _ = random.choice(available_new_targets)
            neighbor_moves[resident_to_change] = new_target_id
            # print(f"动作：更改搬迁 {resident_to_change} -> {new_target_id} (原为 {current_target})")


    return neighbor_moves

# Make sure the previous functions are here, especially:
# load_data, find_potential_moves, calculate_state_metrics, get_neighbor_state

# --- Visualization Functions (可视化函数) ---

def plot_courtyard_layout(df_state, moves, title, filename, empty_courtyards=None, df_info_raw=None):
    """
    绘制庭院布局的示意图。
    Plots a schematic layout of the courtyards.

    Args:
        df_state (pd.DataFrame): DataFrame representing the current state (occupancy).
        moves (dict): Dictionary of moves {resident_id: target_id} for highlighting. None for initial state.
        title (str): Title for the plot.
        filename (str): Filename to save the plot.
        empty_courtyards (list): List of fully emptied courtyard IDs (for final state).
        df_info_raw (pd.DataFrame): Original plot info for details like orientation/area.
    """
    if df_info_raw is None:
         print("错误：绘制布局需要原始地块信息 df_info_raw。")
         print("Error: Plotting layout requires original plot info df_info_raw.")
         return

    courtyards = sorted(df_state['院落ID'].unique())
    num_courtyards = len(courtyards)
    if num_courtyards == 0:
        print("无院落可供绘制。(No courtyards to plot.)")
        return

    # 尝试创建一个大致的网格布局 (Try creating a rough grid layout)
    cols = int(np.ceil(np.sqrt(num_courtyards)))
    rows = int(np.ceil(num_courtyards / cols))
    fig, ax = plt.subplots(figsize=(cols * 4, rows * 4.5)) # Adjust size as needed
    ax.set_aspect('equal', adjustable='box') # Keep aspect ratio

    courtyard_positions = {}
    plot_size = 0.8 # Size of the square representing a plot
    padding = 0.2 # Padding between plots
    courtyard_padding = 2.0 # Padding between courtyards

    max_plots_in_courtyard = df_state.groupby('院落ID').size().max()
    courtyard_box_width = max_plots_in_courtyard * (plot_size + padding) + padding
    courtyard_box_height = (plot_size + padding) # Assume single row display for simplicity now

    current_x, current_y = 0, 0
    for i, cid in enumerate(courtyards):
        courtyard_positions[cid] = (current_x, current_y)
        current_x += courtyard_box_width + courtyard_padding
        if (i + 1) % cols == 0:
            current_x = 0
            current_y -= (courtyard_box_height + courtyard_padding + 1) # Add extra space for title

    plot_details_map = df_info_raw.set_index('地块ID').to_dict('index')
    moved_residents = set(moves.keys()) if moves else set()
    target_plots = set(moves.values()) if moves else set()
    original_resident_plots = df_info_raw[df_info_raw['是否有住户'] == 1]['地块ID'].tolist()

    for cid, (cx, cy) in courtyard_positions.items():
        plots_in_courtyard = df_state[df_state['院落ID'] == cid].sort_values('地块ID')
        num_plots = len(plots_in_courtyard)

        # Draw courtyard boundary
        border_color = 'red' if empty_courtyards and cid in empty_courtyards else 'gray'
        border_lw = 2.5 if empty_courtyards and cid in empty_courtyards else 1
        rect = patches.Rectangle((cx - padding*2, cy - padding*2 - courtyard_box_height),
                                 num_plots * (plot_size + padding) + padding*3,
                                 courtyard_box_height + padding*3 + 1, # Extra height for title
                                 linewidth=border_lw, edgecolor=border_color, facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.text(cx + (num_plots * (plot_size + padding)) / 2, cy + 0.7, f"院落 {cid}",
                ha='center', va='center', fontsize=10, weight='bold')

        plot_x_offset = cx
        for _, plot in plots_in_courtyard.iterrows():
            plot_id = plot['地块ID']
            is_occupied = plot['是否有住户'] == 1
            details = plot_details_map.get(plot_id, {})
            area = details.get('地块面积', '?')
            orient = details.get('地块方位', '?')

            # Determine plot color and label
            face_color = 'lightgray' # Default: empty
            edge_color = 'black'
            label = f"{plot_id}\n{orient} {area}㎡"
            plot_status = "未知 (Unknown)"

            if moves is None: # Initial state
                if plot_id in original_resident_plots:
                    face_color = 'salmon' # Original resident
                    plot_status = "初始住户 (Original Resident)"
                else:
                    face_color = 'lightgray'# Initially empty
                    plot_status = "初始空置 (Initially Empty)"
            else: # Final state
                if plot_id in target_plots:
                    face_color = 'lightblue' # Moved-in resident
                    plot_status = "迁入住户 (Moved-in Resident)"
                elif plot_id in moved_residents: # Original plot of someone who moved
                    face_color = 'lightyellow' # Vacated plot
                    plot_status = "已腾空 (Vacated)"
                elif plot_id in original_resident_plots and plot_id not in moved_residents: # Resident who didn't move
                    face_color = 'salmon'
                    plot_status = "未搬迁住户 (Resident Did Not Move)"
                else: # Was empty, still empty
                    face_color = 'lightgray'
                    plot_status = "始终空置 (Remained Empty)"


            # Draw plot rectangle
            plot_rect = patches.Rectangle((plot_x_offset, cy - courtyard_box_height), plot_size, plot_size,
                                          linewidth=1, edgecolor=edge_color, facecolor=face_color)
            ax.add_patch(plot_rect)
            ax.text(plot_x_offset + plot_size / 2, cy - courtyard_box_height + plot_size / 2, label,
                    ha='center', va='center', fontsize=7)

            plot_x_offset += (plot_size + padding)

    # Create legend manually (粗略图例)
    legend_elements = [
        patches.Patch(facecolor='salmon', edgecolor='black', label='住户 (未搬迁/初始) Resident (Didn\'t move/Initial)'),
        patches.Patch(facecolor='lightblue', edgecolor='black', label='迁入住户 (Moved-in Resident)'),
        patches.Patch(facecolor='lightyellow', edgecolor='black', label='已腾空地块 (Vacated Plot)'),
        patches.Patch(facecolor='lightgray', edgecolor='black', label='空置地块 (Empty Plot)'),
        patches.Patch(facecolor='none', edgecolor='red', linewidth=2, linestyle='--', label='完整空置院落 (Fully Empty Courtyard)')
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


    ax.set_title(title, fontsize=16)
    ax.autoscale_view()
    ax.axis('off') # Hide axes
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    try:
        plt.savefig(filename, dpi=150)
        print(f"布局图已保存至 {filename}")
        print(f"Layout plot saved to {filename}")
    except Exception as e:
        print(f"保存布局图失败: {e}")
        print(f"Failed to save layout plot: {e}")
    plt.close(fig) # Close the figure to free memory


def plot_optimization_progress(history, filename):
    """
    绘制模拟退火过程中的能量（或其他指标）变化。
    Plots the change in energy (or other metrics) during simulated annealing.
    """
    if not history:
        print("无历史记录可供绘制。(No history to plot.)")
        return

    iterations = [item['iteration'] for item in history]
    energy = [item['energy'] for item in history]
    num_empty = [item['num_empty_courtyards'] for item in history] # Example: Track another metric
    costs = [item['total_cost'] for item in history]

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True) # Create 3 subplots sharing x-axis

    # Plot Energy
    axs[0].plot(iterations, energy, label='系统能量 (System Energy)', color='blue')
    axs[0].set_ylabel('能量值 (Energy)')
    axs[0].set_title('模拟退火优化过程 (Simulated Annealing Progress)')
    axs[0].grid(True)
    axs[0].legend()

    # Plot Number of Empty Courtyards
    axs[1].plot(iterations, num_empty, label='完整空置院落数量 (Num Empty Courtyards)', color='green')
    axs[1].set_ylabel('数量 (Count)')
    axs[1].grid(True)
    axs[1].legend()

    # Plot Total Cost
    axs[2].plot(iterations, costs, label='总成本 (Total Cost)', color='red')
    axs[2].axhline(y=BASE_BUDGET, color='orange', linestyle='--', label=f'保底预算 ({BASE_BUDGET:,.0f}) Base Budget')
    axs[2].axhline(y=MAX_BUDGET, color='purple', linestyle=':', label=f'最高预算 ({MAX_BUDGET:,.0f}) Max Budget')
    axs[2].set_xlabel('迭代次数 (Iteration)')
    axs[2].set_ylabel('成本 (Yuan)')
    axs[2].grid(True)
    axs[2].legend()
    axs[2].ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Use scientific notation if numbers are large

    plt.tight_layout()
    try:
        plt.savefig(filename, dpi=150)
        print(f"优化过程图已保存至 {filename}")
        print(f"Optimization progress plot saved to {filename}")
    except Exception as e:
        print(f"保存优化过程图失败: {e}")
        print(f"Failed to save optimization progress plot: {e}")
    plt.close(fig) # Close the figure


def plot_financials(initial_income, final_income, total_cost, filename):
    """
    绘制初始/最终收入对比和成本与预算对比图。
    Plots comparison of initial/final income and cost vs. budget.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5)) # 1 row, 2 columns

    # Income Comparison
    income_labels = ['初始日收入\n(Initial Daily Income)', '最终日收入\n(Final Daily Income)']
    income_values = [initial_income, final_income]
    bars_income = axs[0].bar(income_labels, income_values, color=['gray', 'green'])
    axs[0].set_ylabel('收入 (元/天) Income (Yuan/Day)')
    axs[0].set_title('每日收入对比 (Daily Income Comparison)')
    axs[0].bar_label(bars_income, fmt='{:,.2f}') # Add value labels

    # Cost vs Budget
    cost_labels = ['总成本\n(Total Cost)', '保底预算\n(Base Budget)', '最高预算\n(Max Budget)']
    cost_values = [total_cost, BASE_BUDGET, MAX_BUDGET]
    colors_cost = ['red' if total_cost > MAX_BUDGET else ('orange' if total_cost > BASE_BUDGET else 'blue'), 'orange', 'purple']
    bars_cost = axs[1].bar(cost_labels, cost_values, color=colors_cost)
    axs[1].set_ylabel('金额 (元) Amount (Yuan)')
    axs[1].set_title('总成本与预算 (Total Cost vs Budget)')
    axs[1].bar_label(bars_cost, fmt='{:,.0f}') # Add value labels
    axs[1].ticklabel_format(style='plain', axis='y') # Avoid scientific notation

    plt.tight_layout()
    try:
        plt.savefig(filename, dpi=150)
        print(f"财务总结图已保存至 {filename}")
        print(f"Financial summary plot saved to {filename}")
    except Exception as e:
        print(f"保存财务总结图失败: {e}")
        print(f"Failed to save financial summary plot: {e}")
    plt.close(fig) # Close the figure


# --- Modified Simulated Annealing Function (修改后的模拟退火函数) ---

def simulated_annealing(df_info, adjacency_dict, potential_moves, potential_moves_costs):
    """执行模拟退火算法寻找最优搬迁方案，并记录历史记录 (Execute SA and record history)"""

    # 初始状态：没有搬迁 (Initial state: no moves)
    current_state_moves = {}
    current_metrics = calculate_state_metrics(current_state_moves, df_info, adjacency_dict, potential_moves_costs)
    current_energy = current_metrics['energy']

    best_state_metrics = current_metrics # 记录找到的最佳状态 (Record the best state found)

    temperature = INITIAL_TEMPERATURE

    history = [] # 用于存储历史记录 (For storing history)
    iteration = 0
    # Store initial state in history
    history.append({
        'iteration': iteration,
        'temperature': temperature,
        'energy': current_energy,
        'num_empty_courtyards': current_metrics['num_empty_courtyards'],
        'total_cost': current_metrics['total_cost']
        # Add other metrics if needed for plotting history
    })

    while temperature > MIN_TEMPERATURE:
        accepted_in_temp = 0 # Counter for accepted moves in this temperature step
        for _ in range(MAX_ITERATIONS_PER_TEMP):
            iteration += 1
            # 产生邻近状态 (Generate neighbor state)
            neighbor_moves = get_neighbor_state(current_state_moves, potential_moves, df_info)

            # 计算邻近状态的指标和能量 (Calculate metrics and energy for neighbor state)
            neighbor_metrics = calculate_state_metrics(neighbor_moves, df_info, adjacency_dict, potential_moves_costs)
            neighbor_energy = neighbor_metrics['energy']

            # 计算能量差 (Calculate energy difference)
            delta_energy = neighbor_energy - current_energy

            # 决定是否接受新状态 (Decide whether to accept the new state)
            accept = False
            if delta_energy < 0: # 新状态更好，总是接受 (New state is better, always accept)
                accept = True
            else:
                # 新状态更差，以一定概率接受 (New state is worse, accept with probability)
                acceptance_probability = math.exp(-delta_energy / temperature)
                if random.random() < acceptance_probability:
                    accept = True

            if accept:
                accepted_in_temp += 1
                current_state_moves = neighbor_moves
                current_metrics = neighbor_metrics
                current_energy = neighbor_energy

                # 更新全局最优解 (Update global best solution - same logic as before)
                update_best = False
                if neighbor_metrics['budget_ok']:
                     if not best_state_metrics['budget_ok']: update_best = True
                     else: # Both within budget, compare by priority
                          if neighbor_metrics['num_empty_courtyards'] > best_state_metrics['num_empty_courtyards']: update_best = True
                          elif neighbor_metrics['num_empty_courtyards'] == best_state_metrics['num_empty_courtyards']:
                              if neighbor_metrics['adjacent_empty_pairs'] > best_state_metrics['adjacent_empty_pairs']: update_best = True
                              elif neighbor_metrics['adjacent_empty_pairs'] == best_state_metrics['adjacent_empty_pairs']:
                                  if neighbor_metrics['total_empty_area'] > best_state_metrics['total_empty_area']: update_best = True
                                  elif neighbor_metrics['total_empty_area'] == best_state_metrics['total_empty_area']:
                                      if neighbor_metrics['num_moved'] < best_state_metrics['num_moved']: update_best = True
                                      elif neighbor_metrics['num_moved'] == best_state_metrics['num_moved']:
                                          if neighbor_metrics['total_cost'] < best_state_metrics['total_cost']: update_best = True
                # else: # Handle over-budget comparisons if needed (same as before)

                if update_best:
                    best_state_metrics = copy.deepcopy(neighbor_metrics)
                    # print(f"  Iter {iteration}: 新最优解! ...") # Keep print statement if desired

            # Record history periodically (e.g., every 10 iterations)
            if iteration % 20 == 0: # Adjust frequency as needed
                 history.append({
                     'iteration': iteration,
                     'temperature': temperature,
                     'energy': current_energy, # Record current energy
                     'num_empty_courtyards': current_metrics['num_empty_courtyards'],
                     'total_cost': current_metrics['total_cost']
                 })

        # 降低温度 (Cool down)
        temperature *= COOLING_RATE
        # Optional: Print progress less frequently
        if iteration % (MAX_ITERATIONS_PER_TEMP * 5) == 0: # Print every 5 temp steps
            print(f"Iter {iteration}: Temp: {temperature:.4f}, Accepted: {accepted_in_temp}/{MAX_ITERATIONS_PER_TEMP}, Best Cost: {best_state_metrics['total_cost']:.2f}, Best EmptyC: {best_state_metrics['num_empty_courtyards']}")

    # Add final state to history
    final_metrics = calculate_state_metrics(best_state_metrics['moves_list'], df_info, adjacency_dict, potential_moves_costs)
    history.append({
        'iteration': iteration + 1, # Mark as final iteration
        'temperature': temperature,
        'energy': final_metrics['energy'],
        'num_empty_courtyards': final_metrics['num_empty_courtyards'],
        'total_cost': final_metrics['total_cost']
    })

    print("模拟退火完成。(Simulated Annealing finished.)")
    return best_state_metrics, history # Return history as well


# --- 主程序 (Main Program) ---
if __name__ == "__main__":
    # 1. 加载数据
    df_info_raw, adjacency_dict = load_data(FILE_PATH_INFO, FILE_PATH_ADJACENCY)

    if df_info_raw is not None and adjacency_dict is not None:
        # Create a copy for pristine initial state plotting later
        df_initial_state_vis = df_info_raw.copy()

        # 2. 查找所有可能的搬迁及其成本组成
        potential_moves, df_info_processed = find_potential_moves(df_info_raw.copy()) # Use a copy

        # 创建一个方便查询成本的字典 (Create a dict for easy cost lookup)
        potential_moves_costs = {}
        for res_id, targets in potential_moves.items():
            potential_moves_costs[res_id] = {tar_id: costs for tar_id, costs in targets}

        # 3. 计算初始状态（不搬迁）的收入
        initial_metrics = calculate_state_metrics({}, df_info_processed, adjacency_dict, potential_moves_costs)
        initial_daily_income = initial_metrics['final_daily_income']
        print(f"\n初始状态（未搬迁）下每日总收入: {initial_daily_income:.2f} 元")

        # --- VISUALIZE INITIAL STATE ---
        print("\n正在生成初始布局图...(Generating initial layout plot...)")
        # Need df_info_raw for original details
        plot_courtyard_layout(df_initial_state_vis, moves=None, title="初始院落布局 (Initial Courtyard Layout)", filename="initial_layout.png", df_info_raw=df_initial_state_vis)


        # 4. 运行模拟退火进行优化
        print("\n开始模拟退火优化... (Starting Simulated Annealing optimization...)")
        best_solution_metrics, history = simulated_annealing(df_info_processed.copy(), adjacency_dict, potential_moves, potential_moves_costs) # Pass processed df for calculations

        # 5. 输出最终结果 (Textual Output - same as before)
        print("\n--- 最终搬迁规划结果 (Final Relocation Plan Result) ---")
        # [PREVIOUS CODE: Print results: moves, empty courtyards, cost, income, etc.]
        # [在此处粘贴之前脚本的文本结果输出代码]
        # [PASTE TEXTUAL RESULT OUTPUT CODE HERE]
        if not best_solution_metrics['moves_list']:
            print("根据优化结果，建议不进行任何搬迁。")
            print(f"维持初始状态，每日总收入: {initial_daily_income:.2f} 元")
        else:
            print("【搬迁方案】(Relocation Plan):")
            # ... (rest of the print statements for moves, empty courtyards, cost, income analysis) ...
             # (此处省略之前的详细打印代码，保持不变)
            print(f"共搬迁 {best_solution_metrics['num_moved']} 户居民。")
            resident_details = df_info_processed.set_index('地块ID')
            for resident_id, target_id in best_solution_metrics['moves_list'].items():
                 try:
                     res_courtyard = resident_details.loc[resident_id, '院落ID']
                     tar_courtyard = resident_details.loc[target_id, '院落ID']
                     print(f"  - 院落 {res_courtyard} 的居民 (地块 {resident_id})  搬迁至 -> 院落 {tar_courtyard} 的空地块 {target_id}")
                 except KeyError:
                     print(f"  - 居民地块 {resident_id} 搬迁至 -> 空地块 {target_id} (详细院落信息查找失败)")

            print("\n【腾出的整院】(Emptied Courtyards):")
            if best_solution_metrics['empty_courtyards']:
                print(f"成功腾出 {best_solution_metrics['num_empty_courtyards']} 个完整院落: {best_solution_metrics['empty_courtyards']}")
                print(f"其中，相互毗邻的空置院落对有 {best_solution_metrics['adjacent_empty_pairs']} 对。")
                print(f"腾出的完整院落总面积: {best_solution_metrics['total_empty_area']:.2f} 平方米")
            else:
                print("未能腾出任何完整院落。")

            print("\n【成本与预算】(Cost and Budget):")
            total_cost = best_solution_metrics['total_cost']
            print(f"预计总投入成本: {total_cost:,.2f} 元")
            if total_cost <= BASE_BUDGET:
                print(f"成本在保底规划成本 {BASE_BUDGET:,.2f} 元内。")
            elif total_cost <= MAX_BUDGET:
                print(f"成本在最高预算 {MAX_BUDGET:,.2f} 元内，动用了部分备用金。")
            else:
                print(f"警告：成本 {total_cost:,.2f} 元 超出了最高预算 {MAX_BUDGET:,.2f} 元！该方案可能不可行或需要调整。")

            print("\n【最终收入与盈利】(Final Income and Profit):")
            final_daily_income = best_solution_metrics['final_daily_income']
            income_change = final_daily_income - initial_daily_income
            print(f"搬迁完成后，预计每日总收入: {final_daily_income:,.2f} 元")
            print(f"相比初始状态，每日收入增加: {income_change:,.2f} 元")

            if income_change > 0 and total_cost > 0:
                 payback_days = total_cost / income_change
                 print(f"基于每日收入增加额，预计投资回收期约为: {payback_days:.1f} 天 (约 {payback_days/30:.1f} 个月)")
            elif total_cost > 0 :
                 print("警告：在此搬迁方案下，每日收入未增加或减少，投资无法通过租金增量回收。")
            else: # total_cost == 0 implies no moves were made
                 print("无搬迁成本，收入无变化。")


        # --- VISUALIZE FINAL STATE & PROGRESS ---
        print("\n正在生成最终布局图、优化过程图和财务总结图...(Generating final plots...)")

        # Create DataFrame representing the final state for plotting
        df_final_state_vis = df_info_processed.copy()
        if best_solution_metrics['moves_list']:
             moved_resident_ids = list(best_solution_metrics['moves_list'].keys())
             target_plot_ids = list(best_solution_metrics['moves_list'].values())
             df_final_state_vis.loc[df_final_state_vis['地块ID'].isin(moved_resident_ids), '是否有住户'] = 0
             df_final_state_vis.loc[df_final_state_vis['地块ID'].isin(target_plot_ids), '是否有住户'] = 1

        # Plot Final Layout
        plot_courtyard_layout(df_final_state_vis,
                              moves=best_solution_metrics['moves_list'],
                              title="最终院落布局 (Final Courtyard Layout)",
                              filename="final_layout.png",
                              empty_courtyards=best_solution_metrics['empty_courtyards'],
                              df_info_raw=df_initial_state_vis) # Pass original details

        # Plot Optimization Progress
        plot_optimization_progress(history, filename="optimization_progress.png")

        # Plot Financials
        plot_financials(initial_daily_income,
                        best_solution_metrics['final_daily_income'],
                        best_solution_metrics['total_cost'],
                        filename="financial_summary.png")

        print("\n所有可视化图表已生成。 (All visualization plots generated.)")

    else:
        print("数据加载失败，无法执行程序。")