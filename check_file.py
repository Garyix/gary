import os

target_file = 'data/annotations_10.txt'

print(f"当前工作目录: {os.getcwd()}")
print(f"尝试寻找文件: {target_file}")

if os.path.exists(target_file):
    print("✅ 成功！Python 找到了文件！")
else:
    print("❌ 失败！文件不存在。")
    print("正在检查 data 文件夹里的实际内容...")
    if os.path.exists('data'):
        files = os.listdir('data')
        print(f"data 文件夹里只有这些文件: {files}")
        if 'annotations_10.txt.txt' in files:
            print("⚠️ 发现凶手了！你把文件命名成了 annotations_10.txt.txt (多了一个后缀)")
    else:
        print("❌ 连 data 文件夹都没找到！")