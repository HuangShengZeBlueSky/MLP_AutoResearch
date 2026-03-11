---
layout: home

hero:
  name: "MLP-AutoResearch"
  text: "AI Agent 自主迭代实验框架"
  tagline: "基于 Karpathy's AutoResearch 思想，让 AI 自主探索最优模型架构"
  actions:
    - theme: brand
      text: 📖 方法论 (中文)
      link: /methodology_zh
    - theme: alt
      text: 📖 Methodology (EN)
      link: /methodology_en
    - theme: alt
      text: 查看 GitHub
      link: https://github.com/HuangShengZeBlueSky/MLP_AutoResearch

features:
  - title: 🤖 真正的"自动"研究
    details: 只需设定规则，AI Agent（Claude/GPT）会修改代码、训练、评估，并决定保留或回滚。
  - title: 🎒 教学导向设计
    details: 将原本复杂的模型（如大语言模型）简化为 MNIST 上的多层感知机（MLP），普通 CPU 电脑即可运行。
  - title: 📊 分离关注点基础设施
    details: 明确剥离了不可变的基础评估（prepare.py）和 AI 自由改写的实验空间（train.py）。
---
