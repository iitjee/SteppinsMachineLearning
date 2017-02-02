/*


Softwares required:
Development Kit,
Git,
Gradle, 
IntelliJ,
Apache Common Language,and
Gson.


Explains how you can use java for two components of data science, 
  - data engineering and
  - data analysis. 




IMQAV is a useful framework for data science.
It is an acronym for ingest, model, query, analyze, and visualize. 

Ingestion is a set of software's engineering techniques to adapt high volumes of data that arrive rapidly, often via streaming.
Modeling is a set of data architecture techniques to create data storage that is appropriate for a particular domain. Separating domains into relational key value, etc., these are the products that you can find on the web. 
 Query refers to extracting data from storage, and modifying that data to accommodate anomalies, such as missing data.
 Analyze refers to a set of mathematical techniques for statistics, optimization, and machine learning. 
 Visualize refers to transforming large amounts of data into visually useful formats. Here are some popular programs for visualization.
 
 
 
 We'll be using the Gradle build system, and the Gradle build system is built upon a JVM language, Groovy.
  One interesting component of data engineering is building tools for other data scientists.
  The creators of these JVM languages expanded the idea of building tools into building new languages.
  
  
  Test-Driven Development is a software methodology that emphasizes the role of testing within the software development process.
  Although Test-Driven Development can be considered a separate methodology, most of the principles from Test-Driven 
  Development can easily be incorporated into a light weight process such as Agile, or a complex process such as the
  'Capability Maturity Model'.
  
  The idea is that you and your team create a suite of tests. This suite enables you to be confident of your ability to write
  new code that is compatible with previously written code. Also, you will be confident in your ability to make changes to the 
  previously written code. One benefit of this testing, is that it helps you and your team to catch errors early in the 
  software creation process.
  Another benefit of writing extensive tests, is the creation of executable documentation. 
   In contrast to static documentation, that describes what code does, executable documentation illustrates how code works.
This is valuable for other people who will use your code, and it's also valuable for yourself. On many occasions, I return to code that I wrote a long time ago.
The first thing that I do, is to look at the test code to remind myself of my thought processes while I was originally developing the code.


As an overview there were two types of tests, 
  - black box or functional tests, and 
  - white box or transparent tests.
In black box testing, the people who write the test code and the people who write the applications code are different people, 
often they work on different teams. 
In white box testing, the people who write the test code and the people who write the applications code are the same people.
In this course, we will only discuss white box testing and we will illustrate how to write both test code and applications code.
  
*/
