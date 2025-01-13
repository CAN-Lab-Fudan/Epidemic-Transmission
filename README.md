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
### 2.1 Background Information

In dynamic population analysis and social network research, user contact data based on wireless access points (APs) is a valuable resource. This data could be used to analyze the frequency and duration of user contacts, as well as their temporal and spatial distributions, providing insights into mobility patterns, group behaviors, and social network modeling.
**TeNet** is a MATLAB-based framework for analyzing and processing user contact data. It enables data cleaning, construction of contact records, and statistical analysis of user interactions, ultimately generating contact networks and statistical summaries.
Using this tool, we processed the **FudanWiFi13** dataset, generating several key outputs that can assist researchers in analyzing and understanding user interactions.

### 2.2 Key Features

**TeNet** includes the following core functionalities:

#### 1) Data Import and Preprocessing:
* `openDataSetMain.m`:
  * Import raw data, clean it, and perform unit conversion (e.g., convert contact duration from seconds to minutes).
  * Output formatted data files with userId, startTime, duration, and AP.
* `RecordMain.m`:
  * Filter records based on APs and time ranges.
  * Assign unique user IDs and output the processed user data.

#### 2) Contact Record Construction:
* `PairwiseInteractionNetworkCreator.m`:
  * Construct user contact records based on filtered user activity data.
  * Output user pairs, interaction durations, and timestamps.

#### 3) Contact Duration and Frequency Statistics:
* `StaticIntervalsCreator.m`:
  * Analyze contact records to calculate interaction durations and frequencies.
  * Output aggregated contact data.

#### 4) Contact Information Filtering:
* `PartlyAggregated.m`:
  * Filter specific time ranges or user groups from the contact records.
  * Output filtered contact files.

### 2.3 How to Run
#### 1) Prerequisites:
* MATLAB R2012b or later.
* Input data files in the following format:
  * **Raw data**: e.g., `records.mat` containing `userId`, `startTime`, `duration`, and `location`.
  * **API mapping file**: e.g., `tbaps.txt` mapping access points to users.

#### 2) Setup
1. Clone or download this repository:
```bash
git clone https://github.com/CAN-Lab-Fudan/ViralDynamic/tree/master/TeNet.git
cd contact-network-analysis
```
2. Place the raw data files (`records.mat` and `tbaps.txt`) in the root directory of the project.
3. Open MATLAB and set the current directory to the project folder.

#### 3) Running the Scripts:
Follow these steps to run the tool:

1. **Data Import and Preprocessing**:

   Run `openDataSetMain.m` and `RecordMain.m`:
   ```matlab
   openDataSetMain; % Data cleaning and unit conversion
   RecordMain; % Filtering and user ID assignment
   ```
   Output:
   * `tbdata.mat`: Contains preprocessed user data.

2. **Construct Contact Records**:

   Run `PairwiseInteractionNetworkCreator.m`:
   ```matlab
   PairwiseInteractionNetworkCreator;
   ```
   Output:
   * `tbPairwiseInteraction.txt`: Records user contact information.
   * `tbPairwiseInteraction.mat`: MATLAB file containing contact records.

3. **Contact Duration and Frequency Statistics**:
   Run `StaticIntervalsCreator.m`:
   ```matlab
   StaticIntervalsCreator;
   ```
   Output:
   * `staticInterval.txt`: Detailed contact duration and frequency information.
   * `AggregatedDuration.mat`: Aggregated contact duration statistics.
   * `AggregatedNumber.mat`: Aggregated contact count statistics.

4. **Filter Contact Information**:
   Run ``PartlyAggregated.m``:
   ```matlab
   PartlyAggregated;
   ```
   Output:
   * Filtered contact records based on user-specified criteria.

### 2.4 FudanWiFi13
<img src="https://github.com/CAN-Lab-Fudan/ViralDynamic/blob/master/TeNet/FudanWiFi13/FudanWiFi13_framework.png" width="950px">

Using the provided scripts, we processed the **FudanWiFi13** dataset, containing:
* **12,433 users**
* **1,124,026 login events**
* **206,892 user pairs**
* **2,331,597 interactions**

The generated files are as follows:
1. ``dyadicdata_TimeINDetail.txt``:
   * All records of user pairs.
   * Format: ``User1, User2, StartTime (year, month, day, hour, minute), weekday, duration (min), AP information``.
2. ``singleperson_For_regularSamples_anonymous_TimeINDetail.txt``:
   * All records of individual users.
   * Format: ``User ID, StartTime (year, month, day, hour, minute), weekday, duration (min), AP information``.
3. ``dyadicID.txt``:
   * Mapping of user pair IDs to corresponding users.
   * Format: ``User Pair ID, User1, User2``.
4. ``intervalBeyondDay.txt``:
   * Event time intervals across different days.
   * Format: ``User Pair ID, time_interval (min)``.
5. ``intervalWithinDay.txt``:
   * Event time intervals within the same day.
   * Format: ``User Pair ID, time_interval (min)``.
6. ``dailyNetwork.txt``:
   * Daily network data.
   * Format: ``User1, User2, Day number``.
7. ``aggregated_Network.txt``:
   * Aggregated network node statistics.
   * Format: ``Node (User ID), occurrence count, degree``.

These files provide a comprehensive dataset for researchers to analyze and study.

## 3. SubTrack
### 3.1 Background Information
In modern urban transit systems, understanding passenger travel patterns is critical for improving subway operations and enhancing service quality. The **SubTrack** project simulates passenger trajectories within a subway system based on detailed smart card data. By analyzing passenger card swipe times and matching them to train schedules, the framework reconstructs the specific paths passengers take through the subway network.

This tool provides valuable insights into passenger mobility patterns, travel behavior, and train utilization. By leveraging these insights, transit authorities can optimize subway schedules, enhance passenger experience, and conduct analyses for public health or operational planning. The track_assignment framework offers a robust method for processing and aligning these datasets to support these goals effectively.

### 3.2 Key Features
<img src="https://github.com/CAN-Lab-Fudan/ViralDynamic/blob/master/SubTrack/SubTrack_Framework.png" width="950px">

**SubTrack** includes the following core functionalities:

#### 1) Data Import and Preprocessing:
* **Passenger Data Processing**:
  * Import passenger data from CSV files (e.g., ``stoptime_passengers/{date_id}``).
  * Format data files with columns: Card Number, Passage Time.
  * Perform unit conversion if necessary (e.g., ``timestamps``).
* **Subway Data Processing**:
  * Import subway operation data from CSV files (e.g., ``subways_merged/{week_id}``).
  * Format data files with columns: ``Train Number``, ``Passage Time``.

#### 2) Passenger-Train Matching:
* Match passengers to specific trains based on their passage time and the schedule of subway operations.
* Passenger belongs to Train(i) if:
  
  ``Train(i-1)_Time < Passenger_Time <= Train(i)_Time``
  
* Inputs:
  * ``Train(i-1)/(i)_Time``: Subway passage time from microscope subway data.
  * ``Passenger_Time``: Passenger passage time from microscope passenger data.



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
