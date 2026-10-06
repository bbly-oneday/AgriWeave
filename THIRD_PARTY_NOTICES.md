# 第三方依赖与数据

AgriWeave自有代码采用MIT许可。独立安装的依赖不因本项目许可而改变其许可条件；本仓库不分发第三方原生库、模型或虚拟环境。

主要原生规划依赖为[Fields2Cover](https://github.com/Fields2Cover/Fields2Cover)，采用[BSD-3-Clause](https://github.com/Fields2Cover/Fields2Cover/blob/main/LICENSE)许可。其GDAL、GEOS、OR-tools及其它传递依赖按各自上游许可使用。Python库列表及已验证版本见requirements.txt，各库许可由其上游提供。

研究中使用Fields2Cover时，请同时按[上游引用说明](https://github.com/Fields2Cover/Fields2Cover#citing)引用其论文；AgriWeave的CITATION.cff不是对上游贡献的替代。

人工样例由tempscript/create_demo_fields.py直接构造，不对应真实田块。维护者350田验证的原始数据与结果不包含在本仓库，不能从代码许可推断该数据的再分发许可。
