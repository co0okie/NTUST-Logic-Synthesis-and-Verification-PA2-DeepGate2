import sys
import os
import argparse

# 確保使用者有安裝 matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("錯誤: 找不到 matplotlib 模組。請先執行 'pip install matplotlib'。")
    sys.exit(1)

def parse_log_file(file_path, target_metric, is_stage2=False):
    epochs = []
    values = []
    
    if not os.path.exists(file_path):
        print(f"警告: 找不到檔案 {file_path}，已跳過。")
        return epochs, values

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 切割並清理空白
            parts = [p.strip() for p in line.split('|') if p.strip()]
            
            current_data = {}
            for part in parts:
                if 'epoch:' in part:
                    try:
                        current_data['epoch'] = int(part.split('epoch:')[1].strip())
                    except ValueError:
                        pass
                elif part.startswith('LProb'):
                    current_data['LProb'] = float(part.split()[1])
                elif part.startswith('LRC'):
                    current_data['LRC'] = float(part.split()[1])
                elif part.startswith('LFunc'):
                    current_data['LFunc'] = float(part.split()[1])
                elif part.startswith('loss'):
                    current_data['loss'] = float(part.split()[1])
                elif part.startswith('ACC'):
                    current_data['ACC'] = float(part.split()[1])
            
            # 確保該行同時具備 epoch 以及我們要求的指標
            if 'epoch' in current_data and target_metric in current_data:
                # Stage 1 只有跑機率，沒有 LRC, LFunc, ACC，如果在找這些指標且是 stage1 則跳過
                # if not is_stage2 and target_metric in ['LRC', 'LFunc', 'ACC']:
                #     continue
                
                epochs.append(current_data['epoch'])
                values.append(current_data[target_metric])
                
    return epochs, values

def main():
    # 設定 argparse 來處理命令列參數
    parser = argparse.ArgumentParser(description="繪製 Log 檔案的訓練指標折線圖。")
    parser.add_argument("metric", type=str, choices=['accuracy', 'loss', 'lprob', 'lfunc', 'lrc'],
                        help="要繪製的指標 (例如: accuracy, loss, lprob, lfunc, lrc)")
    
    # 接收剩餘的所有參數，應該是 [Label1, Stage1_Log1, Stage2_Log1, Label2...]
    parser.add_argument("args", nargs=argparse.REMAINDER, 
                        help="每三個一組的標籤與檔案路徑，例如: 'Label1' stage1.log stage2.log")
    
    # 新增可選的輸出路徑參數
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="指定輸出的 PDF 檔案路徑與名稱 (可選)")

    args = parser.parse_args()
    
    # 檢查傳入的標籤與檔案是否為 3 的倍數
    if len(args.args) < 3 or len(args.args) % 3 != 0:
        parser.error("參數必須是每三個一組：<標籤> <Stage1_Log> <Stage2_Log>。")

    metric_arg = args.metric.lower()
    
    # 將使用者輸入的參數對應到 Log 裡的真實 Key
    metric_map = {
        'accuracy': 'ACC',
        'loss': 'loss',    # 這裡指的是驗證集的總 Loss
        'lprob': 'LProb',
        'lfunc': 'LFunc',
        'lrc': 'LRC'
    }

    target_metric = metric_map[metric_arg]
    
    # 擷取 (Label, Stage1_File, Stage2_File) 參數組
    groups = []
    for i in range(0, len(args.args), 3):
        label = args.args[i]
        stage1_file = args.args[i+1]
        stage2_file = args.args[i+2]
        groups.append((label, stage1_file, stage2_file))

    # 開始畫圖設定
    plt.figure(figsize=(10, 6))
    
    # 定義一些不同的標記符號讓折線圖更好辨識
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    has_data = False
    stage1_max_epoch = 0 # 紀錄 Stage1 最大的 epoch，用來畫分隔線
    
    for idx, (label, stage1_file, stage2_file) in enumerate(groups):
        # 讀取 Stage 1 數據
        s1_epochs, s1_values = parse_log_file(stage1_file, target_metric, is_stage2=False)
        # 讀取 Stage 2 數據
        s2_epochs, s2_values = parse_log_file(stage2_file, target_metric, is_stage2=True)
        
        # 串接數據：如果 Stage 1 有數據，Stage 2 的 epoch 要加上 Stage 1 的最大值
        combined_epochs = []
        combined_values = []
        
        if s1_epochs:
            combined_epochs.extend(s1_epochs)
            combined_values.extend(s1_values)
            # 更新分隔線位置
            if max(s1_epochs) > stage1_max_epoch:
                stage1_max_epoch = max(s1_epochs)
        
        if s2_epochs:
            # 位移 Stage 2 的 Epoch
            offset = max(s1_epochs) if s1_epochs else 0
            shifted_s2_epochs = [e + offset for e in s2_epochs]
            
            combined_epochs.extend(shifted_s2_epochs)
            combined_values.extend(s2_values)
            
        if combined_epochs and combined_values:
            has_data = True
            # 如果是 Accuracy，轉換成百分比顯示會更直覺
            if target_metric == 'ACC':
                combined_values = [v * 100 for v in combined_values]
                
            marker = markers[idx % len(markers)]
            plt.plot(combined_epochs, combined_values, marker=marker, linewidth=2, label=label, markersize=6)
        else:
            print(f"警告: 找不到 '{label}' 的有效 {target_metric} 數據。")

    if not has_data:
        print("錯誤: 所有檔案皆無可用數據，無法繪圖。")
        sys.exit(1)

    # 設置圖表細節
    plt.xlabel('Epoch (Stage 1 + Stage 2)', fontsize=16, fontweight='bold')
    
    if target_metric == 'ACC':
        plt.ylabel('Accuracy (%)', fontsize=16, fontweight='bold')
    elif target_metric == 'loss':
        plt.ylabel('Total Loss', fontsize=16, fontweight='bold')
    else:
        plt.ylabel(f'{target_metric} Loss', fontsize=16, fontweight='bold')

    # 加入 Stage 1 與 Stage 2 的垂直分隔線與文字標示
    if stage1_max_epoch > 0:
        plt.axvline(x=stage1_max_epoch + 0.5, color='r', linestyle='--', alpha=0.7)
        # 取得 Y 軸的顯示範圍來放置文字
        ymin, ymax = plt.ylim()
        y_pos = ymin + (ymax - ymin) * 0.05
        
        plt.text(stage1_max_epoch / 2, y_pos, 'Stage 1\n(Prob. only)', 
                 horizontalalignment='center', color='r', fontweight='bold', alpha=0.7,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.5), fontsize=16)
        plt.text(stage1_max_epoch + (max(combined_epochs) - stage1_max_epoch) / 2, y_pos, 'Stage 2\n(Prob. + Func. + RC)', 
                 horizontalalignment='center', color='r', fontweight='bold', alpha=0.7,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.5), fontsize=16)

    # 加入網格線
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 強制 X 軸僅顯示整數 (Epoch 不會有小數)
    plt.gca().xaxis.get_major_locator().set_params(integer=True)

    # 顯示圖例
    plt.legend(loc='best', fontsize=16)
    plt.tight_layout()

    # 決定輸出檔名
    if args.output:
        output_filename = args.output
        # 如果使用者提供的路徑包含資料夾，且該資料夾不存在，則建立它
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_filename = f'{metric_arg}_comparison.pdf'
        
    # 儲存圖片
    plt.savefig(output_filename, format='pdf', bbox_inches='tight')
    print(f"🎉 繪圖成功！圖片已儲存為: {output_filename}")

if __name__ == "__main__":
    main()