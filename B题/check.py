# -*- coding: utf-8 -*-
"""CLAUDE.md 论文合规性检查"""
import re, os

tex = open('main.tex', encoding='utf-8').read()

print('=== 1. 模糊词上下文 ===')
for w in ['约', '大致', '左右', '可能', '或许']:
    for m in re.finditer(w, tex):
        s = max(0, m.start()-15); e = min(len(tex), m.end()+15)
        print(f'  「{w}」: ...{tex[s:e]}...'.replace(chr(10),' '))

print()
print('=== 2. 交叉引用完整性 ===')
labels = re.findall(r'\\label\{([^}]+)\}', tex)
refs = re.findall(r'\\(?:ref|autoref)\{([^}]+)\}', tex)
for l in labels:
    if l not in refs:
        print(f'  ❌ label {l} 未被引用')
for r in set(refs):
    if r not in labels:
        print(f'  ❌ ref {r} 无对应 label')
print('  label 总数:', len(labels), '| ref 总数:', len(refs))

print()
print('=== 3. 图表引用 ===')
figs = re.findall(r'includegraphics\[[^\]]*\]\{output/figures/(\w+)\.png\}', tex)
gen = [f.replace('.png','') for f in os.listdir('output/figures')]
unused = [g for g in gen if g not in figs]
print('  引用图:', len(figs), '个')
print('  生成但未引用:', unused if unused else '无')

print()
print('=== 4. 表格 label 与引用 ===')
tlabels = [l for l in labels if l.startswith('tab:')]
trefs = [r for r in refs if r.startswith('tab:')]
print('  表格 label:', tlabels)
print('  表格 ref:', sorted(set(trefs)))
for l in tlabels:
    if l not in trefs:
        print(f'  ❌ 表格 {l} 未被引用')

print()
print('=== 5. 公式符号使用检查（定义 vs 使用） ===')
# 检查关键符号是否在正文被使用
symbols = ['theta','alpha','alpha_eff','beta','w_deep','w_shallow','eta']
for s in symbols:
    # 粗略统计 math 模式下出现次数
    n = len(re.findall(r'\\'+s.replace('_','_'), tex))
    print(f'  {s}: 出现在公式 {n} 次')

print()
print('=== 6. 全角括号位置 ===')
for m in re.finditer(r'（[^（）]*）', tex):
    s = max(0, m.start()-10); e = min(len(tex), m.end()+10)
    print(f'  {tex[m.start():m.end()]}  ← ...{tex[s:e]}...'.replace(chr(10),' '))
