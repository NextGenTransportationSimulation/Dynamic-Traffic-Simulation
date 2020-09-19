# STALite
 Light-weight computational engine for Static Traffic Assignment on large-scale transportation network
 
 STAlite is an open-source AMS library for efficiently macroscopic traffic assignment 
 based on General Modeling Network Specification (GMNS) format
  
  #What is GMNS?
  General Travel Network Format Specification is a product of Zephyr Foundation, which aims to advance the field through flexible and efficient support, education, guidance, encouragement, and incubation.
  Further Details in https://zephyrtransport.org/projects/2-network-standard-and-tools/
  
  #Goals of STALite development
  1. Provide an open-source code base to enable transportation researchers and software developers to expand its range of capabilities to various traffic management application.
  2. Present results to other users by visualizing time-varying traffic flow dynamics and traveler route choice behavior in an integrated environment.
  3. Provide a free, educational tool for students to understand the complex decision-making process in transportation planning and optimization processes. 

  ![nexta](doc/images/nexta.png)
  
 #Features
 
 1.Network representation based on GMNS format
 
 2.Easy to include demand from different multiple time periods( AM, MD, PM, NT or Hourly)
 
 3.Provide API for both C++ and Python interface 
 
 4.Efficient multi-threading parallel computation and memory management, implemented in C++
 	Utilize up to 40 CPU cores, 200 GB of Memory for networks with more than 50K nodes
 	
 5.Extendable Volume Delay Function (VDF) functions:
   Standard BPR function, and BPR_X function that can obtain dynamic travel time efficiently 
 
  ![STALite](doc/images/output.png)


