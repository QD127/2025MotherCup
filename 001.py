import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 设置 matplotlib 支持中文显示
# 注意：请确保系统中安装了支持中文的字体（例如 SimHei）
# 如果在您的环境中找不到 'SimHei' 字体，请替换为系统中可用的其他中文字体名称
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
except Exception as e:
    print(f"无法设置中文字体 'SimHei'，图形中的中文可能无法正确显示: {e}")
    # 可以尝试其他字体，例如 'Microsoft YaHei', 'WenQuanYi Micro Hei' 等
    # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']


# --- 1. 数据加载与预处理 ---

# 使用提供的示例数据创建 DataFrame
# 实际应用中，请使用 pd.read_excel() 读取您的 Excel 文件
plots_df = pd.read_excel("附件1.xlsx")
adj_df = pd.read_excel("附件2.xlsx")

# plots_df = pd.DataFrame(data_plots)


# adj_df = pd.DataFrame(data_adj)

# --- 数据清洗和准备 ---

# 处理毗邻关系
def parse_adj(adj_str):
    if pd.isna(adj_str) or not isinstance(adj_str, str):
        return []
    # 使用正则表达式提取数字
    ids = re.findall(r'\d+', adj_str)
    return [int(id_str) for id_str in ids]

adj_map = {row['院落ID']: parse_adj(row['毗邻院落号']) for index, row in adj_df.iterrows()}

# 添加采光评分 (南=北=3, 东=2, 西=1)
orientation_scores = {'南': 4, '北': 4, '东': 3, '西': 2}
plots_df['采光评分'] = plots_df['地块方位'].map(orientation_scores)

# 区分住户和空地块
residents_df = plots_df[plots_df['是否有住户'] == 1].copy()
empty_plots_df = plots_df[plots_df['是否有住户'] == 0].copy()

# --- 2. 定义常量和辅助函数 ---

COMMUNICATION_COST = 30000  # 每户沟通成本 (元)
RENOVATION_COST_UPPER = 200000 # 每户修缮成本上限 (元) - 暂定为0，按需调整
RENOVATION_COST_ACTUAL = 0 # 实际使用的修缮成本，贪心策略初始为0
TIME_LOSS_MONTHS = 4       # 每户搬迁耗时 (月)
DAYS_PER_MONTH = 30        # 简化计算，每月天数
DAYS_IN_10_YEARS = 365 * 10 # 十年总天数

# 地块租金 (元/平方米/天)
RENT_RATE = {'南': 15, '北': 15, '东': 8, '西': 8}
plots_df['日租金率'] = plots_df['地块方位'].map(RENT_RATE)

# 整院租金和毗邻加成
WHOLE_COURTYARD_RATE = 30  # 元/平方米/天
ADJACENCY_MULTIPLIER = 1.2

# 计算地块的10年潜在租金
def calculate_plot_10y_rent(area, orientation):
    rate = RENT_RATE.get(orientation, 0)
    return area * rate * DAYS_IN_10_YEARS

# 计算单个院落的10年租金 (考虑是否完整和毗邻)
def calculate_courtyard_10y_rent(courtyard_id, courtyard_area, is_empty, empty_courtyards_set, adj_map):
    if not is_empty:
        return 0 # 非空整院不按整院计算

    base_rent = courtyard_area * WHOLE_COURTYARD_RATE * DAYS_IN_10_YEARS
    is_adjacent_to_empty = False
    adjacent_ids = adj_map.get(courtyard_id, [])
    for adj_id in adjacent_ids:
        if adj_id in empty_courtyards_set:
            is_adjacent_to_empty = True
            break # 只要有一个毗邻空院即可

    if is_adjacent_to_empty:
        # 检查毗邻院落是否也计算了加成，避免双重计算 (如果A邻B，B邻A，只加一次)
        # 简单处理：只要邻，就加成 (实际可能需要更复杂的逻辑确保只加一次)
        return base_rent * ADJACENCY_MULTIPLIER
    else:
        return base_rent

# 计算当前状态的总10年租金收益
def calculate_total_10y_revenue(current_plots_df, current_empty_courtyards, adj_map):
    total_revenue = 0
    processed_courtyards = set()

    # 1. 计算完整空置院落的收益 (包括毗邻加成)
    empty_courtyards_set = set(current_empty_courtyards)
    for cy_id in current_empty_courtyards:
        if cy_id not in processed_courtyards:
            # 获取院落信息 (假设院落面积在plots_df中对应该院落的所有地块都是一致的)
            courtyard_info = current_plots_df[current_plots_df['院落ID'] == cy_id].iloc[0]
            courtyard_area = courtyard_info['院落面积']
            rent = calculate_courtyard_10y_rent(cy_id, courtyard_area, True, empty_courtyards_set, adj_map)
            total_revenue += rent
            processed_courtyards.add(cy_id)
            # 检查并处理毗邻院落，避免重复计算加成 (简化：如果邻居也是空的，它自己计算时会处理)
            # pass

    # 2. 计算非完整院落中空地块的收益
    scattered_empty_plots = current_plots_df[
        (current_plots_df['是否有住户'] == 0) &
        (~current_plots_df['院落ID'].isin(empty_courtyards_set))
    ]
    for index, plot in scattered_empty_plots.iterrows():
        total_revenue += calculate_plot_10y_rent(plot['地块面积'], plot['地块方位'])

    return total_revenue


# 计算单次搬迁的成本
def calculate_move_cost(resident_plot, target_plot):
    cost = 0
    # 沟通成本
    cost += COMMUNICATION_COST
    # 修缮成本 (按策略设定，此处为0)
    cost += RENOVATION_COST_ACTUAL
    # 面积损失成本 (按新地块的租金率计算损失)
    area_diff = target_plot['地块面积'] - resident_plot['地块面积']
    if area_diff > 0:
        area_loss_rent_rate = RENT_RATE.get(target_plot['地块方位'], 0)
        cost += area_diff * area_loss_rent_rate * DAYS_IN_10_YEARS # 10年总损失

    # 时间损失成本 (4个月内，原住地和迁入地均无收益)
    time_loss_days = TIME_LOSS_MONTHS * DAYS_PER_MONTH
    old_plot_rent_loss = resident_plot['地块面积'] * RENT_RATE.get(resident_plot['地块方位'], 0) * time_loss_days
    new_plot_rent_loss = target_plot['地块面积'] * RENT_RATE.get(target_plot['地块方位'], 0) * time_loss_days
    cost += old_plot_rent_loss + new_plot_rent_loss

    return cost

# 检查搬迁是否有效
def is_move_valid(resident_plot, target_plot):
    # 面积补偿: 新 >= 旧, 且 新 <= 1.3 * 旧
    if not (target_plot['地块面积'] >= resident_plot['地块面积'] and \
            target_plot['地块面积'] <= 1.3 * resident_plot['地块面积']):
        return False
    # 采光补偿: 新评分 >= 旧评分
    if not (target_plot['采光评分'] >= resident_plot['采光评分']):
        return False
    return True

# --- 3. 贪心算法模拟 ---

# 初始化状态
current_plots_df = plots_df.copy()
current_residents = current_plots_df[current_plots_df['是否有住户'] == 1].index.tolist()
current_empty_plots = current_plots_df[current_plots_df['是否有住户'] == 0].index.tolist()
courtyard_occupancy = current_plots_df.groupby('院落ID')['是否有住户'].sum()
current_empty_courtyards = courtyard_occupancy[courtyard_occupancy == 0].index.tolist()

# 计算初始状态 (不搬迁) 的10年收益
initial_revenue = calculate_total_10y_revenue(current_plots_df, current_empty_courtyards, adj_map)

history = [] # 记录每一步的状态
total_cost = 0
move_count = 0

# 记录初始状态
history.append({
    'move_count': move_count,
    'total_cost': total_cost,
    'current_revenue': initial_revenue,
    'revenue_increase': 0,
    'm': 0, # 性价比 m
    'move_details': '初始状态',
    'empty_courtyards': list(current_empty_courtyards)
})


while True:
    possible_moves = [] # 存储本轮所有可能的有效搬迁

    # 遍历所有当前住户
    residents_to_consider = current_plots_df.loc[current_residents]
    for resident_idx, resident_plot in residents_to_consider.iterrows():
        # 遍历所有当前空地块
        empty_plots_to_consider = current_plots_df.loc[current_empty_plots]
        for empty_plot_idx, target_plot in empty_plots_to_consider.iterrows():
            # 检查搬迁有效性
            if is_move_valid(resident_plot, target_plot):
                # 预计算成本
                move_cost = calculate_move_cost(resident_plot, target_plot)

                # 预计算搬迁后的状态，判断是否会清空院落
                original_courtyard_id = resident_plot['院落ID']
                residents_in_original_courtyard = current_plots_df[
                    (current_plots_df['院落ID'] == original_courtyard_id) &
                    (current_plots_df['是否有住户'] == 1)
                ].index.tolist()

                empties_courtyard = (len(residents_in_original_courtyard) == 1) # 如果只有这一个住户

                possible_moves.append({
                    'resident_idx': resident_idx,
                    'target_plot_idx': empty_plot_idx,
                    'resident_plot': resident_plot,
                    'target_plot': target_plot,
                    'cost': move_cost,
                    'empties_courtyard': empties_courtyard,
                    'original_courtyard_id': original_courtyard_id
                })

    if not possible_moves:
        print("没有更多有效的搬迁方案了。")
        break # 退出循环

    best_move = None

    # 贪心策略：
    # 1. 优先选择能清空院落的搬迁
    emptying_moves = [m for m in possible_moves if m['empties_courtyard']]
    if emptying_moves:
        # 计算这些搬迁的 "收益/成本" 比
        for move in emptying_moves:
            # 模拟搬迁后的状态来计算收益增量
            temp_plots_df = current_plots_df.copy()
            temp_plots_df.loc[move['target_plot_idx'], '是否有住户'] = 1 # 迁入
            temp_plots_df.loc[move['resident_idx'], '是否有住户'] = 0   # 迁出

            temp_courtyard_occupancy = temp_plots_df.groupby('院落ID')['是否有住户'].sum()
            temp_empty_courtyards = temp_courtyard_occupancy[temp_courtyard_occupancy == 0].index.tolist()

            # 计算搬迁后的总收益
            revenue_after_move = calculate_total_10y_revenue(temp_plots_df, temp_empty_courtyards, adj_map)
            # 计算当前总收益
            current_revenue_before_move = calculate_total_10y_revenue(current_plots_df, current_empty_courtyards, adj_map)

            revenue_gain = revenue_after_move - current_revenue_before_move
            move['revenue_gain'] = revenue_gain
            # 防止除以零错误
            move['gain_cost_ratio'] = revenue_gain / move['cost'] if move['cost'] > 0 else np.inf

        # 选择收益成本比最高的搬迁
        emptying_moves.sort(key=lambda x: x['gain_cost_ratio'], reverse=True)
        best_move = emptying_moves[0]

    else:
        # 2. 如果没有能清空院落的，选择成本最低的搬迁
        possible_moves.sort(key=lambda x: x['cost'])
        best_move = possible_moves[0]

    # 执行最佳搬迁
    if best_move:
        move_count += 1
        res_idx = best_move['resident_idx']
        tar_idx = best_move['target_plot_idx']
        move_cost = best_move['cost']

        # 更新地块状态
        current_plots_df.loc[tar_idx, '是否有住户'] = 1
        current_plots_df.loc[res_idx, '是否有住户'] = 0

        # 更新住户和空地列表
        current_residents.remove(res_idx)
        current_empty_plots.remove(tar_idx)
        current_empty_plots.append(res_idx)

        # 更新总成本
        total_cost += move_cost

        # 重新计算当前空置院落
        courtyard_occupancy = current_plots_df.groupby('院落ID')['是否有住户'].sum()
        current_empty_courtyards = courtyard_occupancy[courtyard_occupancy == 0].index.tolist()

        # 计算当前总收益
        current_revenue = calculate_total_10y_revenue(current_plots_df, current_empty_courtyards, adj_map)
        revenue_increase = current_revenue - initial_revenue

        # 计算性价比 m
        m = revenue_increase / total_cost if total_cost > 0 else 0

        # 记录历史
        move_details = f"第{move_count}步: 居民地块 {res_idx}(院落{best_move['resident_plot']['院落ID']}) -> 空地块 {tar_idx}(院落{best_move['target_plot']['院落ID']})"
        history.append({
            'move_count': move_count,
            'total_cost': total_cost,
            'current_revenue': current_revenue,
            'revenue_increase': revenue_increase,
            'm': m,
            'move_details': move_details,
            'empty_courtyards': list(current_empty_courtyards),
            'resident_moved_from': res_idx,
            'resident_moved_to': tar_idx,
        })
        print(f"{move_details}, 成本: {move_cost:.2f}, 累计成本: {total_cost:.2f}, 累计收益增量: {revenue_increase:.2f}, m: {m:.4f}")

    else:
        # 如果 best_move 为 None (理论上在 possible_moves 非空时不应发生)
        print("未能确定最佳搬迁方案。")
        break


# --- 4. 拐点分析 ---

history_df = pd.DataFrame(history)

turning_point_info = None
max_m = 0
m_threshold = 20 # 性价比阈值

# 寻找 m >= threshold 且 m 开始下降或增幅显著减缓的点
# 简单策略：找到 m 首次达到顶峰（且>=threshold）的位置
peak_m_idx = -1
for i in range(1, len(history_df)):
    current_m = history_df.loc[i, 'm']
    prev_m = history_df.loc[i-1, 'm']
    if current_m > max_m:
        max_m = current_m
        # 只有当 m >= threshold 时才考虑作为潜在峰值
        if max_m >= m_threshold:
             peak_m_idx = i # 记录当前峰值的位置

    # 如果 m 开始下降，并且之前的峰值满足条件，则认为找到了拐点
    if current_m < prev_m and peak_m_idx != -1 and history_df.loc[peak_m_idx, 'm'] >= m_threshold :
         # 拐点是峰值点，即 peak_m_idx
         turning_point_info = history_df.loc[peak_m_idx]
         print(f"\n找到性价比拐点 (m >= {m_threshold} 且之后开始下降):")
         print(f"发生在第 {turning_point_info['move_count']} 次搬迁后。")
         break
    # 另一种情况：如果m的增量变得非常小（例如小于某个epsilon），也可以认为是拐点
    # 这里简化处理，只看下降


# 如果循环结束还没找到下降的拐点，但存在 m >= threshold 的点，取 m 最大的点
if turning_point_info is None and peak_m_idx != -1:
     turning_point_info = history_df.loc[peak_m_idx]
     print(f"\n性价比 m 在模拟过程中持续增长或保持最高值，未明显下降。")
     print(f"性价比最高点 (m >= {m_threshold}) 发生在第 {turning_point_info['move_count']} 次搬迁后。")

elif turning_point_info is None:
    print(f"\n在模拟过程中，性价比 m 未能达到 {m_threshold}。")
    if not history_df.empty and history_df['m'].max() > 0:
        max_m_point = history_df.loc[history_df['m'].idxmax()]
        print(f"性价比 m 的最大值为 {max_m_point['m']:.4f}，发生在第 {max_m_point['move_count']} 次搬迁后。")
        print(f"若要存在拐点，可能需要降低性价比阈值 m，或者调整搬迁策略/成本计算。")
    else:
        print("没有进行任何搬迁或性价比始终为0。")


# 输出拐点详细信息
if turning_point_info is not None:
    print("\n--- 拐点状态详情 ---")
    print(f"搬迁次数: {turning_point_info['move_count']}")
    print(f"累计投入成本: {turning_point_info['total_cost']:.2f} 元")
    print(f"达到的性价比 m: {turning_point_info['m']:.4f}")
    print(f"腾出的整院ID列表: {turning_point_info['empty_courtyards']}")

    # 计算拐点时的整院面积和最终收入/盈利
    final_empty_courtyards = turning_point_info['empty_courtyards']
    final_empty_courtyards_set = set(final_empty_courtyards)
    final_plots_state_df = plots_df.copy() # 需要根据 history 恢复到拐点时的状态

    # 恢复拐点时的地块状态 (需要从头模拟到拐点)
    temp_plots_df_at_tp = plots_df.copy()
    moves_at_tp = history_df.iloc[1:turning_point_info.name + 1] # 获取从第一次到拐点的所有搬迁记录
    for idx, move_rec in moves_at_tp.iterrows():
        res_idx_tp = move_rec['resident_moved_from']
        tar_idx_tp = move_rec['resident_moved_to']
        if pd.notna(res_idx_tp) and pd.notna(tar_idx_tp): # 确保不是初始状态行
             temp_plots_df_at_tp.loc[int(tar_idx_tp), '是否有住户'] = 1
             temp_plots_df_at_tp.loc[int(res_idx_tp), '是否有住户'] = 0


    final_total_revenue = calculate_total_10y_revenue(temp_plots_df_at_tp, final_empty_courtyards, adj_map)
    final_profit = final_total_revenue - initial_revenue - turning_point_info['total_cost'] # 盈利 = 10年总收益 - 初始10年收益 - 搬迁成本

    total_empty_courtyard_area = 0
    processed_cy_area = set()
    for cy_id in final_empty_courtyards:
         if cy_id not in processed_cy_area:
            # 从原始数据（或拐点状态数据）获取院落面积
            area_info = temp_plots_df_at_tp[temp_plots_df_at_tp['院落ID'] == cy_id]
            if not area_info.empty:
                 total_empty_courtyard_area += area_info.iloc[0]['院落面积']
                 processed_cy_area.add(cy_id)


    print(f"最终腾出整院总面积: {total_empty_courtyard_area:.2f} 平方米")
    print(f"最终状态下10年总租金收益: {final_total_revenue:.2f} 元")
    print(f"对比初始状态的10年盈利 (收益增量 - 成本): {final_profit:.2f} 元")

    print("\n搬迁详情 (截至拐点):")
    # 找到拐点对应的搬迁记录
    moves_df = history_df.iloc[1:turning_point_info.name+1] # 从1开始，排除初始状态
    for index, row in moves_df.iterrows():
         print(row['move_details'])


# --- 5. 可视化 ---
# --- 之前的代码保持不变 ---
# ... (包括数据加载、预处理、常量定义、函数定义、贪心算法模拟、history_df 的生成) ...

# --- 4. 拐点分析 (这部分可以保留计算，但我们将主要输出条件改为 m < 20) ---
history_df = pd.DataFrame(history)

# --- (原拐点寻找逻辑可以注释掉或移除，因为我们关注的是 m < 20 的点) ---
# turning_point_info = None
# max_m = 0
# m_threshold = 20 # 性价比阈值
# peak_m_idx = -1
# ... (原拐点寻找代码) ...
# if turning_point_info is not None:
#     print("\n--- 拐点状态详情 ---")
#     ... (原拐点输出代码) ...


# --- 5. 可视化 (可以保留，用于观察整体趋势) ---
if not history_df.empty and len(history_df) > 1:
    fig, ax2 = plt.subplots(figsize=(12, 6))
    color = 'tab:blue'
    ax2.set_ylabel('性价比 m', color=color)
    # 过滤掉无效的 m 值 (inf, NaN) 以便绘图
    valid_m = history_df[np.isfinite(history_df['m'])]
    ax2.plot(valid_m['move_count'], valid_m['m'], color=color, marker='x', linestyle='--', label='性价比 m')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(y=20, color='gray', linestyle=':', label='m=20 参考线')
    ax2.set_xlabel('搬迁次数') # x轴标签

    # 标记 m 首次低于 20 的点（可选，可以在图上标出触发输出的点）
    # (需要找到 index_when_dropped，见下面逻辑)

    fig.tight_layout()
    plt.title('搬迁过程中性价比(m)变化趋势')
    # 使用 ax2.legend() 避免 fig.legend() 可能的警告或问题
    ax2.legend(loc='best')
    plt.grid(True) # 添加网格线
    plt.show()
else:
    print("\n无法生成可视化图表，因为没有足够的搬迁数据。")


# --- 寻找并输出 m 首次低于 20 前的状态 ---
state_before_m_drop_info = None
found_drop_point = False
index_when_dropped = -1 # 记录 m<20 发生的行的索引

if len(history_df) > 1:
    for i in range(1, len(history_df)): # 从第二次记录开始检查
        current_m = history_df.loc[i, 'm']
        # 确保 m 是有效的数值再比较
        if pd.notna(current_m) and np.isfinite(current_m) and current_m < 20:
            # 找到了首次低于 20 的点，我们需要记录它 *前一步* 的状态
            if i > 0: # 确保存在前一步 (i=0是初始状态)
                state_before_m_drop_info = history_df.loc[i-1] # 获取前一行的信息
                index_when_dropped = i # 记录是哪一步之后 m < 20
                found_drop_point = True
                print(f"\n检测到：第 {i} 次搬迁后，性价比 m ({current_m:.4f}) 首次低于 20。")
                print(f"将输出第 {i-1} 次搬迁后的状态信息 (m 首次低于 20 之前)。")
                break # 找到第一个就停止
            else:
                # 如果第一次搬迁后 m 就 < 20
                print(f"\n注意：第 1 次搬迁后，性价比 m ({current_m:.4f}) 即低于 20。")
                # 此时没有 "前一步" 的搬迁后状态，可以考虑输出初始状态或不输出
                # 这里我们选择不进入输出环节，因为没有 "m < 20 之前" 的搬迁后状态
                found_drop_point = False # 重置标志，不触发后续输出
                break

if not found_drop_point and len(history_df) > 1:
    print("\n在模拟过程中，性价比 m 始终保持在 20 或以上，或从未计算出有效的m值。")

# --- 如果找到了 m < 20 的点，并且成功记录了前一步状态，则输出 ---
if found_drop_point and state_before_m_drop_info is not None:
    print("\n--- 状态详情 (m 首次低于 20 之前) ---")
    print(f"搬迁次数: {state_before_m_drop_info['move_count']}")
    print(f"累计投入成本: {state_before_m_drop_info['total_cost']:.2f} 元")
    print(f"达到的性价比 m: {state_before_m_drop_info['m']:.4f}")

    # 获取该状态下的空院落列表
    final_empty_courtyards = state_before_m_drop_info['empty_courtyards']
    if isinstance(final_empty_courtyards, str): # 如果存储的是字符串形式的列表
        try:
            final_empty_courtyards = eval(final_empty_courtyards)
        except:
            print("警告：无法解析空院落列表字符串。")
            final_empty_courtyards = []
    elif not isinstance(final_empty_courtyards, list):
         final_empty_courtyards = [] # 确保是列表

    print(f"腾出的整院ID列表: {final_empty_courtyards}")

    # --- 需要重新构建此时的地块状态来计算面积和收益 ---
    plots_df_at_state = plots_df.copy() # 从原始状态开始
    # 获取截至 "m低于20之前" 的所有搬迁记录 (从第1次到 state_before_m_drop_info 对应的次数)
    # state_before_m_drop_info.name 是其在 history_df 中的索引
    moves_to_state = history_df.iloc[1 : state_before_m_drop_info.name + 1]

    for idx, move_rec in moves_to_state.iterrows():
        res_idx = move_rec['resident_moved_from']
        tar_idx = move_rec['resident_moved_to']
        # 确保索引有效且存在于DataFrame中
        if pd.notna(res_idx) and pd.notna(tar_idx):
             res_idx_int = int(res_idx)
             tar_idx_int = int(tar_idx)
             if res_idx_int in plots_df_at_state.index and tar_idx_int in plots_df_at_state.index:
                plots_df_at_state.loc[tar_idx_int, '是否有住户'] = 1
                plots_df_at_state.loc[res_idx_int, '是否有住户'] = 0
             else:
                 print(f"警告: 搬迁记录 {idx} 的索引 {res_idx} 或 {tar_idx} 在原始数据中找不到，跳过此步状态更新。")


    # --- 使用构建好的状态计算指标 ---
    final_total_revenue_state = calculate_total_10y_revenue(plots_df_at_state, final_empty_courtyards, adj_map)
    # 盈利 = 此状态下的总收益 - 初始收益 - 此状态下的总成本
    final_profit_state = final_total_revenue_state - initial_revenue - state_before_m_drop_info['total_cost']

    # 计算腾出整院总面积
    total_empty_courtyard_area_state = 0
    processed_cy_area_state = set()
    for cy_id in final_empty_courtyards:
         if cy_id not in processed_cy_area_state:
            # 从 *此刻状态* 的DataFrame获取院落面积信息
            area_info = plots_df_at_state[plots_df_at_state['院落ID'] == cy_id]
            if not area_info.empty:
                 # 假设院落面积在院落内各地块记录一致
                 total_empty_courtyard_area_state += area_info.iloc[0]['院落面积']
                 processed_cy_area_state.add(cy_id)

    print(f"最终腾出整院总面积: {total_empty_courtyard_area_state:.2f} 平方米")
    print(f"最终状态下10年总租金收益: {final_total_revenue_state:.2f} 元")
    print(f"对比初始状态的10年盈利 (收益增量 - 成本): {final_profit_state:.2f} 元")

    print("\n搬迁详情 (截至m首次低于20之前):")
    if state_before_m_drop_info['move_count'] > 0:
        # moves_to_state 已经包含了正确的搬迁记录
        for index, row in moves_to_state.iterrows():
             print(row['move_details'])
    else:
        print("无搬迁记录 (初始状态)。")

elif len(history_df) <= 1:
    print("\n未进行搬迁或只有初始状态，无法判断 m 是否低于 20。")

# --- 代码结束 ---
