# ML-Salted-Fish-Cultivation
咸鱼养成日记
# 1.Logistic Regression问题记录

## 1.1.问题介绍
&emsp;&emsp;模型效果很大程度上取决于learning rate，即alpha。起初未引入动态调整alpha策略，只有SGD参数优化方法对应的模型表现比较好；引入动态
调整alhpa策略后，MBDG参数优化方法对应的模型表现有所改善，BGD效果仍然很差。
    
## 1.2.问题分析
&emsp;&emsp;经过分析后，造成上述情况的原因为alpha：若不采用动态调整alpha策略，由于BGD每次更新参数采用全部数据，步幅较大，会导致后期参数不收敛，而MBGD由于采用batch策略，稍微缓解了这种弊端，SGD同理。<br/>
&emsp;&emsp;引入动态更新策略后，初期效果仍然不好的原因为：随着迭代次数增加，alpha衰减过快，导致模型未能很好地进行迭代学习。
    
## 1.3.解决策略
&emsp;&emsp;解决策略为:合理设置alpha衰减策略，既要解决收敛问题，又不能过快衰减。
