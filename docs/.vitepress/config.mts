import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "MLP-AutoResearch",
  description: "AI Agent auto-iterates MLP experiments on MNIST",
  base: "/MLP_AutoResearch/",
  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: '方法论 (ZH)', link: '/methodology_zh' },
      { text: 'Methodology (EN)', link: '/methodology_en' },
      { text: '实验结果 / Results', link: '/results' }
    ],

    sidebar: [
      {
        text: 'Documentation / 文档',
        items: [
          { text: '📖 方法论分析 (中文)', link: '/methodology_zh' },
          { text: '📖 Methodology (English)', link: '/methodology_en' },
          { text: '📊 实验结果 / Results', link: '/results' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/HuangShengZeBlueSky/MLP_AutoResearch' }
    ],
    
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2026'
    }
  }
})
