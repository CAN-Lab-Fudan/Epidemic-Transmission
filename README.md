# ViralDynamics
<img src="https://github.com/CAN-Lab-Fudan/ViralDynamics/blob/master/logo.png" width="180px">

## 1. Introduction
**ViralDynamics** is a Python framework specifically developed for the simulation and analysis of epidemic spreading on complex networks. The framework combines several essential functionalities, including temporal network construction, trajectory planning, higher-order dynamics prediction, propagation dynamics simulation, and immunization game evolution analysis.. This tutorial aims to help you get started. By completing this tutorial, you will discover the core functionalities of **ViralDynamics**. 

Key features of **ViralDynamics** include:
1. **TeNet**: A temporal network modeling tool for large-scale WiFi data. Allows users to create dynamic networks that represent interactions within a population, simulating disease spread across different types of connections.

2. **SubTrack**: A trajectory planning tool for massive metro swipe card data. Supports users in obtaining precise individual trajectories during metro trips to analyze mobility patterns.

3. **TaHiP**: A high-order dynamics prediction tool. Allows users to accurately predict higher-order dynamic with unknown higher-order topology.

4. **ε-SIS**: A tool for simulating ε-susceptible-infected-susceptible (SIS) dynamics in continuous time.

5. **EvoVax**: A social perception-based immunization game evolution analysis tool. 

## 2. TeNet
* Background Information

In dynamic population analysis and social network research, user contact data based on wireless access points (APs) is a valuable resource. This data could be used to analyze the frequency and duration of user contacts, as well as their temporal and spatial distributions, providing insights into mobility patterns, group behaviors, and social network modeling.
**TeNet** is a MATLAB-based framework for analyzing and processing user contact data. It enables data cleaning, construction of contact records, and statistical analysis of user interactions, ultimately generating contact networks and statistical summaries.

* Key Features

**TeNet** includes the following core functionalities:


一、标题写法：
第一种方法：
1、在文本下面加上 等于号 = ，那么上方的文本就变成了大标题。等于号的个数无限制，但一定要大于0个哦。。
2、在文本下面加上 下划线 - ，那么上方的文本就变成了中标题，同样的 下划线个数无限制。
3、要想输入=号，上面有文本而不让其转化为大标题，则需要在两者之间加一个空行。
另一种方法：（推荐这种方法；注意⚠️中间需要有一个空格）
关于标题还有等级表示法，分为六个等级，显示的文本大小依次减小。不同等级之间是以井号  #  的个数来标识的。一级标题有一个 #，二级标题有两个# ，以此类推。
例如：
# 一级标题  
## 二级标题  
### 三级标题  
#### 四级标题  
##### 五级标题  
###### 六级标题 
二、编辑基本语法  
1、字体格式强调
 我们可以使用下面的方式给我们的文本添加强调的效果
*强调*  (示例：斜体)  
 _强调_  (示例：斜体)  
**加重强调**  (示例：粗体)  
 __加重强调__ (示例：粗体)  
***特别强调*** (示例：粗斜体)  
___特别强调___  (示例：粗斜体)  
2、代码  
`<hello world>`  
3、代码块高亮  
```
@Override
protected void onDestroy() {
    EventBus.getDefault().unregister(this);
    super.onDestroy();
}
```  
4、表格 （建议在表格前空一行，否则可能影响表格无法显示）
 
 表头  | 表头  | 表头
 ---- | ----- | ------  
 单元格内容  | 单元格内容 | 单元格内容 
 单元格内容  | 单元格内容 | 单元格内容  
 
5、其他引用
图片  
![图片名称](https://www.baidu.com/img/bd_logo1.png)  
链接  
[链接名称](https://www.baidu.com/)    
6、列表 
1. 项目1  
2. 项目2  
3. 项目3  
   * 项目1 （一个*号会显示为一个黑点，注意⚠️有空格，否则直接显示为*项目1） 
   * 项目2   
 
7、换行（建议直接在前一行后面补两个空格）
直接回车不能换行，  
可以在上一行文本后面补两个空格，  
这样下一行的文本就换行了。
或者就是在两行文本直接加一个空行。
也能实现换行效果，不过这个行间距有点大。  
 
8、引用
> 第一行引用文字  
> 第二行引用文字   
