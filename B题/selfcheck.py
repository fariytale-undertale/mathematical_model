# -*- coding: utf-8 -*-
"""论文深度自检"""
import re

tex = open('main.tex', encoding='utf-8').read()

print('=== 1. 表格竖线检查（booktabs 规范：无竖线）===')
bad = 0
for m in re.finditer(r'\\begin\{tabular\}\{([^}]*)\}', tex):
    spec = m.group(1)
    if '|' in spec:
        print(f'  ⚠️ 含竖线: {{{spec}}}')
        bad += 1
print('  结论:', '全部无竖线 ✓' if bad == 0 else f'{bad} 处含竖线')

print()
print('=== 2. 摘要字数 ===')
# 提取摘要部分（标题到关键词）
m = re.search(r'摘要\}(.*?)关键词', tex, re.S)
if m:
    abs_text = m.group(1)
    # 去掉 latex 命令，粗略统计中文字符
    abs_text_clean = re.sub(r'\$[^$]*\$', '', abs_text)  # 去公式
    abs_text_clean = re.sub(r'\\[a-zA-Z]+', '', abs_text_clean)  # 去命令
    abs_text_clean = re.sub(r'[{}]', '', abs_text_clean)
    cn = len(re.findall(r'[\u4e00-\u9fff]', abs_text_clean))
    print(f'  摘要中文字符数(含公式占位): {cn}')
    print('  建议 800-1000 字（含标点符号则更多）')

print()
print('=== 3. 数字一致性：摘要 vs 正文关键数字 ===')
for key in ['170.27', '315.71', '34', '68', '49', '245', '4.21', '94.8', '416.55', '415.69']:
    n = tex.count(key)
    print(f'  「{key}」出现 {n} 次')

print()
print('=== 4. 括号数量 ===')
paren = len(re.findall(r'（[^（）]*）', tex))
print(f'  全角括号 {paren} 对')

print()
print('=== 5. 连续出现多个括号的句子检查 ===')
# 按句号/换行切分，找含 >=3 对括号的句子
sentences = re.split(r'[。\n]', tex)
for s in sentences:
    if len(re.findall(r'（[^（）]*）', s)) >= 3:
        print('  ⚠️ 一句含≥3对括号:', s.strip()[:80], '...')

print()
print('=== 6. 表格 row 数检查（粒度：8-15行最佳）===')
for m in re.finditer(r'\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}', tex, re.S):
    body = m.group(1)
    nrows = body.count('\\\\')  # 行分隔
    if nrows > 20:
        print(f'  ⚠️ 表格 {nrows} 行（偏多）')

print()
print('=== 7. 空行/重复空格等排版问题 ===')
if '  ' in tex and '\\begin' not in tex:
    pass
# 检查是否有 "——" 中文破折号（应保留）
print('  中文破折号 —— 出现', tex.count('—'), '次（用于表1首行，正常）')

print()
print('=== 8. \% 与 % 混用检查 ===')
pct_bad = len(re.findall(r'[^\\]%\s', tex))
print(f'  未转义的 % 号（可能破坏注释）: {pct_bad}')
