//  Portions Copyright 2019
// Xuesong Zhou
//   If you help write or modify the code, please also list your names here.
//   The reason of having Copyright info here is to ensure all the modified version, as a whole, under the GPL 
//   and further prevent a violation of the GPL.

// More about "How to use GNU licenses for your own software"
// http://www.gnu.org/licenses/gpl-howto.html

#pragma warning( disable : 4305 4267 4018) 
#include <iostream>
#include <fstream>
#include <list> 
#include <omp.h>
#include <algorithm>
#include <time.h>
#include <functional>
#include <stdio.h>   
#include <math.h>


#include <stack>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <iostream>
#include <iomanip>
using namespace std;
using std::string;
using std::ifstream;
using std::vector;
using std::map;
using std::istringstream;
using std::max;
template <typename T>

// some basic parameters setting

//Pls make sure the _MAX_K_PATH > Agentlite.cpp's g_number_of_K_paths+g_reassignment_number_of_K_paths and the _MAX_ZONE remain the same with .cpp's defination
#define _MAX_LABEL_COST 9999999999999999999999.0

#define _MAX_AGNETTYPES 4 //because of the od demand store format,the MAX_demandtype must >=g_DEMANDTYPES.size()+1
#define _MAX_TIMEPERIODS 1
#define _MAX_LINK_SIZE_IN_A_PATH 1000
#define _MAX_LINK_SIZE_FOR_A_NODE 200

#define _MAX_TIMESLOT_PerPeriod 100 // max 96 15-min slots per day

#define MIN_PER_TIMESLOT 15

// Linear congruential generator 
#define LCG_a 17364
#define LCG_c 0
#define LCG_M 65521  // it should be 2^32, but we use a small 16-bit number to save memory


#define sprintf_s sprintf

FILE* g_pFileOutputLog = NULL;
int g_debugging_flag = 0;

#define STRING_LENGTH_PER_LINE 20000

void fopen_ss(FILE **file, const char *fileName, const char *mode)
{
	*file = fopen(fileName, mode);
}


void g_ProgramStop();


//below shows where the functions used in Agentlite.cpp come from!
//Utility.cpp

#pragma warning(disable: 4244)  // stop warning: "conversion from 'int' to 'float', possible loss of data"


class CCSVParser
{
public:
	char Delimiter;
	bool IsFirstLineHeader;
	ifstream inFile;
	string mFileName;
	vector<string> LineFieldsValue;
	vector<string> Headers;
	map<string, int> FieldsIndices;

	vector<int> LineIntegerVector;

public:
	void  ConvertLineStringValueToIntegers()
	{
		LineIntegerVector.clear();
		for (unsigned i = 0; i < LineFieldsValue.size(); i++)
		{
			std::string si = LineFieldsValue[i];
			int value = atoi(si.c_str());

			if (value >= 1)
				LineIntegerVector.push_back(value);

		}
	}
	vector<string> GetHeaderVector()
	{
		return Headers;
	}

	bool m_bDataHubSingleCSVFile;
	string m_DataHubSectionName;
	bool m_bLastSectionRead;

	bool m_bSkipFirstLine;  // for DataHub CSV files

	CCSVParser(void)
	{
		Delimiter = ',';
		IsFirstLineHeader = true;
		m_bSkipFirstLine = false;
		m_bDataHubSingleCSVFile = false;
		m_bLastSectionRead = false;
	}

	~CCSVParser(void)
	{
		if (inFile.is_open()) inFile.close();
	}


	bool OpenCSVFile(string fileName, bool b_required)
	{
		mFileName = fileName;
		inFile.open(fileName.c_str());

		if (inFile.is_open())
		{
			if (IsFirstLineHeader)
			{
				string s;
				std::getline(inFile, s);
				vector<string> FieldNames = ParseLine(s);

				for (size_t i = 0;i < FieldNames.size();i++)
				{
					string tmp_str = FieldNames.at(i);
					size_t start = tmp_str.find_first_not_of(" ");

					string name;
					if (start == string::npos)
					{
						name = "";
					}
					else
					{
						name = tmp_str.substr(start);
						//			TRACE("%s,", name.c_str());
					}


					FieldsIndices[name] = (int)i;
				}
			}

			return true;
		}
		else
		{
			if (b_required)
			{

				cout << "File " << fileName << " does not exist. Please check." << endl;
				//g_ProgramStop();
			}
			return false;
		}
	}


	void CloseCSVFile(void)
	{
		inFile.close();
	}



	bool ReadRecord()
	{
		LineFieldsValue.clear();

		if (inFile.is_open())
		{
			string s;
			std::getline(inFile, s);
			if (s.length() > 0)
			{

				LineFieldsValue = ParseLine(s);

				return true;
			}
			else
			{

				return false;
			}
		}
		else
		{
			return false;
		}
	}

	vector<string> ParseLine(string line)
	{
		vector<string> SeperatedStrings;
		string subStr;

		if (line.length() == 0)
			return SeperatedStrings;

		istringstream ss(line);


		if (line.find_first_of('"') == string::npos)
		{

			while (std::getline(ss, subStr, Delimiter))
			{
				SeperatedStrings.push_back(subStr);
			}

			if (line.at(line.length() - 1) == ',')
			{
				SeperatedStrings.push_back("");
			}
		}
		else
		{
			while (line.length() > 0)
			{
				size_t n1 = line.find_first_of(',');
				size_t n2 = line.find_first_of('"');

				if (n1 == string::npos && n2 == string::npos) //last field without double quotes
				{
					subStr = line;
					SeperatedStrings.push_back(subStr);
					break;
				}

				if (n1 == string::npos && n2 != string::npos) //last field with double quotes
				{
					size_t n3 = line.find_first_of('"', n2 + 1); // second double quote

					//extract content from double quotes
					subStr = line.substr(n2 + 1, n3 - n2 - 1);
					SeperatedStrings.push_back(subStr);

					break;
				}

				if (n1 != string::npos && (n1 < n2 || n2 == string::npos))
				{
					subStr = line.substr(0, n1);
					SeperatedStrings.push_back(subStr);
					if (n1 < line.length() - 1)
					{
						line = line.substr(n1 + 1);
					}
					else // comma is the last char in the line string, push an empty string to the back of vector
					{
						SeperatedStrings.push_back("");
						break;
					}
				}

				if (n1 != string::npos && n2 != string::npos && n2 < n1)
				{
					size_t n3 = line.find_first_of('"', n2 + 1); // second double quote
					subStr = line.substr(n2 + 1, n3 - n2 - 1);
					SeperatedStrings.push_back(subStr);
					size_t idx = line.find_first_of(',', n3 + 1);

					if (idx != string::npos)
					{
						line = line.substr(idx + 1);
					}
					else
					{
						break;
					}
				}
			}

		}

		return SeperatedStrings;
	}

	template <class T> bool GetValueByFieldName(string field_name, T& value, bool NonnegativeFlag = true)
	{


		bool required_field = true;
		bool print_out = false;
		if (FieldsIndices.find(field_name) == FieldsIndices.end())
		{
			if (required_field)
			{
				cout << "Field " << field_name << " in file " << mFileName << " does not exist. Please check the file." << endl;

				g_ProgramStop();
			}
			return false;
		}
		else
		{
			if (LineFieldsValue.size() == 0)
			{
				return false;
			}

			int size = (int)(LineFieldsValue.size());
			if (FieldsIndices[field_name] >= size)
			{
				return false;
			}

			string str_value = LineFieldsValue[FieldsIndices[field_name]];

			if (str_value.length() <= 0)
			{
				return false;
			}

			istringstream ss(str_value);

			T converted_value;
			ss >> converted_value;

			if (/*!ss.eof() || */ ss.fail())
			{
				return false;
			}

			if (NonnegativeFlag && converted_value < 0)
				converted_value = 0;

			value = converted_value;
			return true;
		}
	}


	bool GetValueByFieldName(string field_name, string& value)
	{
		if (FieldsIndices.find(field_name) == FieldsIndices.end())
		{
			return false;
		}
		else
		{
			if (LineFieldsValue.size() == 0)
			{
				return false;
			}

			unsigned int index = FieldsIndices[field_name];
			if (index >= LineFieldsValue.size())
			{
				return false;
			}
			string str_value = LineFieldsValue[index];

			if (str_value.length() <= 0)
			{
				return false;
			}

			value = str_value;
			return true;
		}
	}

};



template <typename T>
T **AllocateDynamicArray(int nRows, int nCols)
{
	T **dynamicArray;

	dynamicArray = new (std::nothrow) T*[nRows];

	if (dynamicArray == NULL)
	{
		cout << "Error: insufficient memory.";
		g_ProgramStop();

	}

	for (int i = 0; i < nRows; i++)
	{
		dynamicArray[i] = new (std::nothrow) T[nCols];

		if (dynamicArray[i] == NULL)
		{
			cout << "Error: insufficient memory.";
			g_ProgramStop();
		}


	}

	return dynamicArray;
}

template <typename T>
void DeallocateDynamicArray(T** dArray, int nRows, int nCols)
{
	if (!dArray)
		return;

	for (int x = 0; x < nRows; x++)
	{
		delete[] dArray[x];
	}

	delete[] dArray;

}

template <typename T>
T ***Allocate3DDynamicArray(int nX, int nY, int nZ)
{
	T ***dynamicArray;

	dynamicArray = new (std::nothrow) T**[nX];

	if (dynamicArray == NULL)
	{
		cout << "Error: insufficient memory.";
		g_ProgramStop();
	}

	for (int x = 0; x < nX; x++)
	{
		if (x % 1000 == 0)
		{
			cout << "allocating 3D memory for " << x << endl;
		}


		dynamicArray[x] = new (std::nothrow) T*[nY];

		if (dynamicArray[x] == NULL)
		{
			cout << "Error: insufficient memory.";
			g_ProgramStop();
		}

		for (int y = 0; y < nY; y++)
		{
			dynamicArray[x][y] = new (std::nothrow) T[nZ];
			if (dynamicArray[x][y] == NULL)
			{
				cout << "Error: insufficient memory.";
				g_ProgramStop();
			}
		}
	}

	for (int x = 0; x < nX; x++)
		for (int y = 0; y < nY; y++)
			for (int z = 0; z < nZ; z++)
			{
				dynamicArray[x][y][z] = 0;
			}
	return dynamicArray;

}

template <typename T>
void Deallocate3DDynamicArray(T*** dArray, int nX, int nY)
{
	if (!dArray)
		return;
	for (int x = 0; x < nX; x++)
	{
		for (int y = 0; y < nY; y++)
		{
			delete[] dArray[x][y];
		}

		delete[] dArray[x];
	}

	delete[] dArray;

}



template <typename T>
T ****Allocate4DDynamicArray(int nM, int nX, int nY, int nZ)
{
	T ****dynamicArray;

	dynamicArray = new (std::nothrow) T***[nX];

	if (dynamicArray == NULL)
	{
		cout << "Error: insufficient memory.";
		g_ProgramStop();
	}
	for (int m = 0; m < nM; m++)
	{
		if (m % 100 == 0)
			cout << "allocating 4D memory for " << m << " zones" << endl;

		dynamicArray[m] = new (std::nothrow) T**[nX];

		if (dynamicArray[m] == NULL)
		{
			cout << "Error: insufficient memory.";
			g_ProgramStop();
		}

		for (int x = 0; x < nX; x++)
		{
			dynamicArray[m][x] = new (std::nothrow) T*[nY];

			if (dynamicArray[m][x] == NULL)
			{
				cout << "Error: insufficient memory.";
				g_ProgramStop();
			}

			for (int y = 0; y < nY; y++)
			{
				dynamicArray[m][x][y] = new (std::nothrow) T[nZ];
				if (dynamicArray[m][x][y] == NULL)
				{
					cout << "Error: insufficient memory.";
					g_ProgramStop();
				}
			}
		}
	}
	return dynamicArray;

}

template <typename T>
void Deallocate4DDynamicArray(T**** dArray, int nM, int nX, int nY)
{
	if (!dArray)
		return;
	for (int m = 0; m < nM; m++)
	{
		for (int x = 0; x < nX; x++)
		{
			for (int y = 0; y < nY; y++)
			{
				delete[] dArray[m][x][y];
			}

			delete[] dArray[m][x];
		}
		delete[] dArray[m];
	}
	delete[] dArray;

}


//struct MyException : public exception {
//	const char * what() const throw () {
//		return "C++ Exception";
//	}
//};
//

class CDemand_Period {
public:

	CDemand_Period()
	{
		demand_period_id = 0;
		starting_time_slot_no = 0;
		ending_time_slot_no = 0;

	}
	string demand_period;
	string time_period;
	int demand_period_id;
	int starting_time_slot_no;
	int ending_time_slot_no;

	int get_time_horizon_in_min()
	{
		return (ending_time_slot_no - starting_time_slot_no) * 15;
	}

};


class CAgent_type {
public:
	CAgent_type()
	{
		value_of_time = 1;
		agent_type_no = 0;
		flow_type = 0;
	}

	int agent_type_no;
	string agent_type;
	float value_of_time;  // dollar per hour
	std::map<int, float> PCE_link_type_map;  // link type, product consumption equivalent used, for travel time calculation
	std::map<int, float> CRU_link_type_map;  // link type, 	Coefficient of Resource Utilization - CRU, for resource constraints 
	int flow_type; // not enter the column_pool optimization process. 0: continuous, 1: fixed, 2 discrete.

};

class CLinkType
{
public:
	int link_type;
	string link_type_name;
	string agent_type_list;
	string type_code;

	int number_of_links;

	CLinkType()
	{
		number_of_links = 0;
		link_type = 1;
	}

	bool AllowAgentType(string agent_type)
	{
		if (agent_type_list.size() == 0)  // if the agent_type_list is empty then all types are allowed.
			return true;
		else
		{
			if (agent_type_list.find(agent_type) != string::npos)  // otherwise, only an agent type is listed in this "white list", then this agent is allowed to travel on this link
				return true;
			else
				return false;


		}
	}


};


class CColumnPath {
public:
int* path_node_vector;
int* path_link_vector;

int m_node_size;
int m_link_size;

void AllocateVector(int node_size, int* node_vector, int link_size, int * link_vector)
{
	m_node_size = node_size;
	m_link_size = link_size;
	path_node_vector = new int[node_size];  // dynamic array
	path_link_vector = new int[link_size];

	for (int i = 0; i < m_node_size; i++)  // copy backward
	{
		path_node_vector[i] = node_vector[m_node_size - 1 - i];
	}

	for (int i = 0; i < m_link_size; i++)
	{
		path_link_vector[i] = link_vector[m_link_size - 1- i];
	}


}


int path_seq_no;

float path_volume;  // path volume
float path_switch_volume;  // path volume
float path_travel_time;
float path_distance;
float path_cost;
float path_gradient_cost;  // first order graident cost.
float path_gradient_cost_difference;  // first order graident cost - least gradient cost
float path_gradient_cost_relative_difference;  // first order graident cost - least gradient cost


CColumnPath()
{
	path_node_vector = NULL;
	path_link_vector = NULL;

	path_switch_volume = 0;
	path_seq_no = 0;
	path_cost = 0;
	path_volume = 0;
	path_travel_time = 0;
	path_distance = 0;
	path_gradient_cost = 0;
	path_gradient_cost_difference = 0;
	path_gradient_cost_relative_difference = 0;
}

~CColumnPath()
{
	if (m_node_size >= 1)
	{ 
		delete path_node_vector;
		delete path_link_vector;
	}

}
};

class CAgentPath {
public:
	int path_id;
	int o_node_no;
	int d_node_no;
	float travel_time;
	float distance;
	float volume;
	int node_sum;
	
	std::vector <int> path_link_sequence;

	CAgentPath()
	{
		path_id = 0;
		node_sum = -1;

		travel_time = 0;
		distance = 0;
		volume = 0;
	}
};
class CColumnVector {
public:
	float cost;
	float time;
	float distance;
	float od_volume;  // od volume

	std::vector <CAgentPath>  discrete_agent_path_vector;  // first key is the sum of node id;. e.g. node 1, 3, 2, sum of those node ids is 6, 1, 4, 2 then node sum is 7.

	std::map <int, CColumnPath> path_node_sequence_map;  // first key is the sum of node id;. e.g. node 1, 3, 2, sum of those node ids is 6, 1, 4, 2 then node sum is 7.
	// this is colletion of unique paths
	CColumnVector()
	{
		od_volume = 0;
		cost = 0;
		time = 0;
		distance = 0;
	}
};

class CAgent_Column {
public:
	
	
	int agent_id;
	int o_zone_id;
	int d_zone_id;
	int o_node_id;
	int d_node_id;
	string agent_type;
	string demand_period;
	float volume;	
	float cost;
	float travel_time;
	float distance;
	vector<int> path_node_vector;
	vector<int> path_link_vector;
	vector<float> path_time_vector;

	CAgent_Column()
	{
		cost = 0;
	}


};

class Assignment {
public:
	Assignment()
	{
		g_number_of_memory_blocks = 300;
		total_demand_volume = 0.0; 
		g_column_pool = NULL;
		g_origin_demand_array = NULL;
		//pls check following 7 settings before running programmer
		g_number_of_threads = 4000;
		g_number_of_K_paths = 20;
		g_number_of_demand_periods = 24;
		g_reassignment_tau0 = 999;

		g_number_of_links = 0;
		g_number_of_nodes = 0;
		g_number_of_zones = 0;
		g_number_of_agent_types = 0;

		b_debug_detail_flag = 1;

		g_pFileDebugLog = NULL;
		assignment_mode = 0;  // default is UE
	}

	void InitializeDemandMatrix(int number_of_zones, int number_of_agent_types, int number_of_time_periods)
	{
		g_number_of_zones = number_of_zones;
		g_number_of_agent_types = number_of_agent_types;

		g_column_pool = Allocate4DDynamicArray<CColumnVector>(number_of_zones, number_of_zones, max(1, number_of_agent_types), number_of_time_periods);
		g_origin_demand_array = Allocate3DDynamicArray<float>(number_of_zones, max(1, number_of_agent_types), number_of_time_periods);


		for (int i = 0;i < number_of_zones;i++)
		{
			for (int at = 0;at < number_of_agent_types;at++)
			{
				for (int tau = 0;tau < g_number_of_demand_periods;tau++)
				{

					g_origin_demand_array[i][at][tau] = 0.0;
				}
			}

		}
		total_demand_volume = 0.0;
		for (int i = 0;i < number_of_agent_types;i++)
		{
			for (int tau = 0;tau < g_number_of_demand_periods;tau++)
			{
				total_demand[i][tau] = 0.0;
			}
		}

		g_DemandGlobalMultiplier = 1.0f;


	};
	~Assignment()
	{

		if (g_column_pool != NULL)
			Deallocate4DDynamicArray(g_column_pool, g_number_of_zones, g_number_of_zones, g_number_of_agent_types);
		if (g_origin_demand_array != NULL)
			Deallocate3DDynamicArray(g_origin_demand_array, g_number_of_zones, g_number_of_agent_types);

		if (g_pFileDebugLog != NULL)
			fclose(g_pFileDebugLog);

	}
	int g_number_of_threads;
	int g_number_of_K_paths;
	int assignment_mode;
	int g_number_of_memory_blocks;

	int g_reassignment_tau0;

	int b_debug_detail_flag;
	std::map<int, int> g_internal_node_to_seq_no_map;  // hash table, map external node number to internal node sequence no. 
	std::map<int, int> g_zoneid_to_zone_seq_no_mapping;// from integer to integer map zone_id to zone_seq_no



	CColumnVector**** g_column_pool;
	float*** g_origin_demand_array;

	//StatisticOutput.cpp
	float total_demand_volume;
	//NetworkReadInput.cpp and ShortestPath.cpp



	std::vector<CDemand_Period> g_DemandPeriodVector;
	std::vector<CAgent_type> g_AgentTypeVector;
	std::map<int, CLinkType> g_LinkTypeMap;

	std::map<string, int> demand_period_to_seqno_mapping;
	std::map<string, int> agent_type_2_seqno_mapping;


	float total_demand[_MAX_AGNETTYPES][_MAX_TIMEPERIODS];
	float g_DemandGlobalMultiplier;

	int g_number_of_links;
	int g_number_of_nodes;
	int g_number_of_zones;
	int g_number_of_agent_types;
	int g_number_of_demand_periods;

	FILE* g_pFileDebugLog = NULL;

};

Assignment assignment;

class CVDF_Period
{
public:

	CVDF_Period()
	{
		m = 0.5;
		VOC = 0;
		gamma = 3.47f;
		mu = 1000;
		theta = 1;
		alpha = 0.15f;
		beta = 4;
		rho = 1;
		marginal_base = 1;
		ruc_base_resource = 0;

		ruc_type = 0;

		starting_time_slot_no = 0;
		ending_time_slot_no = 0;
	}


	int starting_time_slot_no;  // in 15 min slot
	int ending_time_slot_no;
	string period;


	//standard BPR parameter 
	float alpha;
	float beta;
	float theta;
	float capacity;
	float FFTT;
	float VOC;
	float rho;
	float ruc_base_resource;
	int   ruc_type;

	float marginal_base;
	//updated BPR-X parameters
	float gamma;
	float mu;
	float m;
	float congestion_period_P;
	// inpput
	float volume;

	//output
	float avg_delay;
	float avg_travel_time = 0;
	float avg_waiting_time = 0;

	//float Q[_MAX_TIMESLOT_PerPeriod];  // t starting from starting_time_slot_no if we map back to the 24 hour horizon 
	float waiting_time[_MAX_TIMESLOT_PerPeriod];
	float arrival_rate[_MAX_TIMESLOT_PerPeriod];

	float discharge_rate[_MAX_TIMESLOT_PerPeriod];
	float travel_time[_MAX_TIMESLOT_PerPeriod];

	
	float get_waiting_time(int relative_time_slot_no)
	{
		if (relative_time_slot_no >=0 && relative_time_slot_no < _MAX_TIMESLOT_PerPeriod)
			return waiting_time[relative_time_slot_no];
		else
			return 0;

	}
	int t0, t3;

	void Setup()
	{

	}

	float  PerformBPR(float volume)
	{
		volume = max(0, volume);  // take nonnegative values

		VOC = volume / max(0.00001f, capacity);
		avg_travel_time = FFTT * theta + FFTT * alpha * pow(volume / max(0.00001f, capacity), beta);

		marginal_base = FFTT * alpha * beta*pow(volume / max(0.00001f, capacity), beta - 1);



	return avg_travel_time;

		// volume --> avg_traveltime
	
	}

	
	float PerformBPR_X(float volume)
	{
		congestion_period_P = 0;
		// Step 1: Initialization
		int L = ending_time_slot_no - starting_time_slot_no;  // in 15 min slot

		if (L >= _MAX_TIMESLOT_PerPeriod - 1)
			return 0;

		float mid_time_slot_no = starting_time_slot_no + L / 2.0;  // t1;
		for (int t = 0; t <= L; t++)
		{
			waiting_time[t] = 0;
			arrival_rate[t] = 0;
			discharge_rate[t]= mu/2.0;
			travel_time[t] = FFTT;
		}
		avg_waiting_time = 0;
		avg_travel_time = FFTT + avg_waiting_time;

		//int L = ending_time_slot_no - starting_time_slot_no;  // in 15 min slot

		// Case 1
		if (volume <= L * mu / 2)
		{
			// still keep 0 waiting time for all time period
			congestion_period_P = 0;

		}
		else
		{
			//if (volume > L * mu / 2 ) // Case 2
			float P = min(L, volume * 2 / mu - L);  // if  volume > L * mu then P is set the the maximum of L: case 3
			congestion_period_P = P/4.0;  // unit: hour

			t0 = mid_time_slot_no - P / 2.0;
			t3 = mid_time_slot_no + P / 2.0;
			int t2 = m * (t3 - t0) + t0;
			for (int tt = 0; tt <= L; tt++)
			{
				int time = starting_time_slot_no + tt;
				if (time < t0)
				{
					waiting_time[tt] = 0;
					arrival_rate[tt] = mu / 2;
					discharge_rate[tt] = mu / 2.0;
					travel_time[tt] = FFTT;

				}
				if (time >= t0 && time <= t3)
				{
					waiting_time[tt] = 1 / (4.0*mu) *gamma *(time - t0)*(time - t0) * (time - t3)*(time - t3);
					arrival_rate[tt] = gamma * (time - t0)*(time - t2)*(time - t3) + mu;
					discharge_rate[tt] = mu;
					travel_time[tt] = FFTT + waiting_time[tt];
				}
				if (time > t3)
				{
					waiting_time[tt] = 0;
					arrival_rate[tt] = mu / 2;
					discharge_rate[tt] = mu / 2.0;
					travel_time[tt] = FFTT;
				}
				avg_waiting_time = gamma / (120 * mu)*pow(P, 4.0);
				//cout << avg_waiting_time << endl;
				avg_travel_time = FFTT + avg_waiting_time;
			}
		}

		return avg_travel_time;

	}
};


class CLink
{
public:
	CLink()  // construction 
	{
		zone_seq_no_for_outgoing_connector = -1;

		free_flow_travel_time_in_min = 1;
		cost = 0;
		for (int tau = 0; tau < _MAX_TIMEPERIODS; tau++)
		{
			flow_volume_per_period[tau] = 0;
			resource_per_period[tau] = 0;

			queue_length_perslot[tau] = 0;
			travel_time_per_period[tau] = 0;

			for(int at = 0; at < _MAX_AGNETTYPES; at++)
			{
				volume_per_period_per_at[tau][at] = 0;
				resource_per_period_per_at[tau][at] = 0;

			}

			TDBaseTT[tau] = 0;
			TDBaseCap[tau] = 0;
			TDBaseFlow[tau] = 0;
			TDBaseQueue[tau] = 0;


			//cost_perhour[tau] = 0;
		}
		link_spatial_capacity = 100;
		RUC_type = 0;
	}

	~CLink()
	{
		//if (flow_volume_for_each_o != NULL)
		//	delete flow_volume_for_each_o;
	}

	void free_memory()
	{
	}

	void AddAgentsToLinkVolume()
	{


	}



	// 1. based on BPR. 

	int zone_seq_no_for_outgoing_connector ;
	int m_LeftTurn_link_seq_no;

	int m_RandomSeed;
	int link_seq_no;
	int link_id;
	int from_node_seq_no;
	int to_node_seq_no;
	int link_type;
	float cost;

	float PCE_at[_MAX_AGNETTYPES];
	float CRU_at[_MAX_AGNETTYPES];


	float fftt;
	float free_flow_travel_time_in_min;


	CVDF_Period VDF_period[_MAX_TIMEPERIODS];

	float TDBaseTT[_MAX_TIMEPERIODS];
	float TDBaseCap[_MAX_TIMEPERIODS];
	float TDBaseFlow[_MAX_TIMEPERIODS];
	float TDBaseQueue[_MAX_TIMEPERIODS];


	int type;
	float link_spatial_capacity;

	//static
	//float flow_volume;
	//float travel_time;

	float flow_volume_per_period[_MAX_TIMEPERIODS];
	float volume_per_period_per_at[_MAX_TIMEPERIODS][_MAX_AGNETTYPES];

	float resource_per_period[_MAX_TIMEPERIODS];
	float resource_per_period_per_at[_MAX_TIMEPERIODS][_MAX_AGNETTYPES];

	float queue_length_perslot[_MAX_TIMEPERIODS];  // # of vehicles in the vertical point queue
	float travel_time_per_period[_MAX_TIMEPERIODS];
	float travel_marginal_cost_per_period[_MAX_TIMEPERIODS][_MAX_AGNETTYPES];

	float exterior_penalty_cost_per_period[_MAX_TIMEPERIODS][_MAX_AGNETTYPES];
	float exterior_penalty_derivative_per_period[_MAX_TIMEPERIODS][_MAX_AGNETTYPES];

	int RUC_type;
	int number_of_periods;

	float length;
	//std::vector <SLinkMOE> m_LinkMOEAry;
	//beginning of simulation data 

	//toll related link
	//int m_TollSize;
	//Toll *pTollVector;  // not using SLT here to avoid issues with OpenMP

	void CalculateTD_VDFunction();

	float get_VOC_ratio(int tau)
	{

		return (flow_volume_per_period[tau] + TDBaseFlow[tau]) / max(0.00001f, TDBaseCap[tau]);
	}

	float get_net_resource(int tau)
	{

		return resource_per_period[tau];
	}

	float get_speed(int tau)
	{
		return length / max(travel_time_per_period[tau], 0.0001f) * 60;  // per hour
	}


	void calculate_marginal_cost_for_agent_type(int tau, int agent_type_no, float PCE_agent_type)
	{
		// volume * dervative 
		// BPR_term: volume * FFTT * alpha * (beta) * power(v/c, beta-1), 
		
		travel_marginal_cost_per_period[tau][agent_type_no] = VDF_period[tau].marginal_base*PCE_agent_type;
	}

	void calculate_penalty_for_agent_type(int tau, int agent_type_no, float CRU_agent_type)
	{
		// volume * dervative 
		if (RUC_type == 0)
			return;

		float resource = 0;
			
		if (RUC_type == 2)  // equality constraints 
			resource = resource_per_period[tau];
		else
			resource = min(0, resource_per_period[tau]);  // inequality
		

		exterior_penalty_derivative_per_period[tau][agent_type_no] = 2*VDF_period[tau].rho * resource * CRU_agent_type ;

		if (exterior_penalty_derivative_per_period[tau][agent_type_no] < -100)
		{
			int i_debug = 1;
		}

	}

	void calculate_Gauss_Seidel_penalty_for_agent_type(int tau, int agent_type_no, float CRU_agent_type)
	{
		// volume * dervative 
		if (RUC_type == 0)
			return;

		float resource = 0;  // resource in _Gauss_Seidel framework is refereed to all the other resourcs used by others

		if (RUC_type == 2)  // equality constraints 
			resource = resource_per_period[tau];
		else
			resource = min(0, resource_per_period[tau]);  // inequality


		exterior_penalty_derivative_per_period[tau][agent_type_no] =  VDF_period[tau].rho * (2* resource * CRU_agent_type + CRU_agent_type);

	}

	float get_generalized_first_order_gradient_cost_of_second_order_loss_for_agent_type(int tau, int agent_type_no)
	{

		float generalized_cost = travel_time_per_period[tau] + cost / assignment.g_AgentTypeVector[agent_type_no].value_of_time * 60;  // *60 as 60 min per hour

		if (assignment.assignment_mode == 2 || assignment.assignment_mode == 3)  // system optimal mode or exterior panalty mode
		{
			generalized_cost += travel_marginal_cost_per_period[tau][agent_type_no];
		}

		if ((assignment.assignment_mode) == 2 && (RUC_type != 0) )  // exterior panalty mode
		{
			generalized_cost += exterior_penalty_derivative_per_period[tau][agent_type_no];

		}

		return generalized_cost;
	}



};


class CNode
{
public:
	CNode()
	{
		zone_id = -1;
		node_seq_no = -1;
		//accessible_node_count = 0;
	}

	//int accessible_node_count;

	int node_seq_no;  // sequence number 
	int node_id;      //external node number 
	int zone_id = -1;

	double x;
	double y;

	std::vector<int> m_outgoing_link_seq_no_vector;
	std::vector<int> m_to_node_seq_no_vector;
	std::map<int, int> m_to_node_seq_no_map;

};


extern std::vector<CNode> g_node_vector;
extern std::vector<CLink> g_link_vector;


class COZone
{
public:
	int zone_seq_no;  // 0, 1, 
	int zone_id;  // external zone id // this is origin zone
	int node_seq_no;


};

extern std::vector<COZone> g_zone_vector;
extern std::map<int, int> g_zoneid_to_zone_seq_no_mapping;

class CAGBMAgent
{
public:

	int agent_id;
	int income;
	int gender;
	int vehicle;
	int purpose;
	int flexibility;
	float preferred_arrival_time;
	float travel_time_in_min;
	float free_flow_travel_time;
	int from_zone_seq_no;
	int to_zone_seq_no;
	int type;
	int time_period;
	int k_path;
	float volume;
	float arrival_time_in_min;



};
extern std::vector<CAGBMAgent> g_agbmagent_vector;

struct CNodeForwardStar{
	int* OutgoingLinkNoArray;
	int* OutgoingNodeNoArray;
	int OutgoingLinkSize;
};

class NetworkForSP  // mainly for shortest path calculation
{
public:
	bool m_bSingleSP_Flag;
	
	NetworkForSP()
	{
		m_value_of_time = 10;
		m_memory_block_no = 0;

		temp_path_node_vector_size = 1000;
		
		temp_path_node_vector = new int[temp_path_node_vector_size];
		temp_path_link_vector = new int[temp_path_node_vector_size];

	}

	int temp_path_node_vector_size;
	int* temp_path_node_vector; //node seq vector for each ODK
	int* temp_path_link_vector; //node seq vector for each ODK

	int m_memory_block_no;

	std::vector<int>  m_origin_node_vector; // assigned nodes for computing 
	std::vector<int>  m_origin_zone_seq_no_vector;
	int  tau; // assigned nodes for computing 
	int  m_agent_type_no; // assigned nodes for computing 
	float m_value_of_time;


	CNodeForwardStar* NodeForwardStarArray;

	int m_threadNo;  // internal thread number 

	int m_ListFront; // used in coding SEL
	int m_ListTail;  // used in coding SEL
	int* m_SENodeList; // used in coding SEL

	float* m_node_label_cost;  // label cost // for shortest path calcuating
	float* m_label_time_array;  // time-based cost
	float* m_label_distance_array;  // distance-based cost

	int * m_node_predecessor;  // predecessor for nodes
	int* m_node_status_array; // update status 
	int* m_link_predecessor;  // predecessor for this node points to the previous link that updates its label cost (as part of optimality condition) (for easy referencing)

	float* m_link_flow_volume_array; 


	// major function 1:  allocate memory and initialize the data 
	void AllocateMemory(int number_of_nodes, int number_of_links)
	{
		NodeForwardStarArray = new CNodeForwardStar[number_of_nodes]; 

		m_SENodeList = new int[number_of_nodes];  //1
		m_node_status_array = new int[number_of_nodes];  //2
		m_label_time_array = new float[number_of_nodes];  //3
		m_label_distance_array = new float[number_of_nodes];  //4
		m_node_predecessor = new int[number_of_nodes];  //5
		m_link_predecessor = new int[number_of_nodes];  //6
		m_node_label_cost = new float[number_of_nodes];  //7

		m_link_flow_volume_array = new float[number_of_links];

	}

	void BuildNetwork(Assignment& assignment)
	{
		int m_outgoing_link_seq_no_vector[_MAX_LINK_SIZE_FOR_A_NODE];
		int m_to_node_seq_no_vector[_MAX_LINK_SIZE_FOR_A_NODE];

	
		for (int i = 0; i < assignment.g_number_of_nodes; i++) //Initialization for all non-origin nodes
		{
			int outgoing_link_size = 0;
	
			for (int j = 0; j < g_node_vector[i].m_outgoing_link_seq_no_vector.size(); j++)
			{
			
				int link_seq_no = g_node_vector[i].m_outgoing_link_seq_no_vector[j];

				
				if(assignment.g_LinkTypeMap[g_link_vector[link_seq_no].link_type].AllowAgentType (assignment.g_AgentTypeVector[m_agent_type_no].agent_type ))  // only predefined allowed agent type can be considered
				{ 
					m_outgoing_link_seq_no_vector[outgoing_link_size] = link_seq_no;
					m_to_node_seq_no_vector[outgoing_link_size] = g_node_vector[i].m_to_node_seq_no_vector[j];

					outgoing_link_size++;

					if (outgoing_link_size >= _MAX_LINK_SIZE_FOR_A_NODE)
					{
						cout << " Error: outgoing_link_size >= _MAX_LINK_SIZE_FOR_A_NODE" << endl;
						g_ProgramStop();
					}
					
				}

			}
			//// add outgoing link data into dynamic array 
			int node_seq_no = g_node_vector[i].node_seq_no;
			NodeForwardStarArray[node_seq_no].OutgoingLinkSize = outgoing_link_size;
			if(outgoing_link_size>=1)
			{ 
			NodeForwardStarArray[node_seq_no].OutgoingLinkNoArray = new int[outgoing_link_size];
			NodeForwardStarArray[node_seq_no].OutgoingNodeNoArray = new int[outgoing_link_size];
			}


			for (int l = 0; l < outgoing_link_size; l++)
			{
				NodeForwardStarArray[node_seq_no].OutgoingLinkNoArray[l] = m_outgoing_link_seq_no_vector[l];
				NodeForwardStarArray[node_seq_no].OutgoingNodeNoArray[l] = m_to_node_seq_no_vector[l];
			}
					
		}


		m_value_of_time = assignment.g_AgentTypeVector[m_agent_type_no].value_of_time;
	}
	



	~NetworkForSP()
	{

		for (int i = 0; i < assignment.g_number_of_nodes; i++) //Initialization for all non-origin nodes
		{

			if (NodeForwardStarArray[i].OutgoingLinkSize > 0)
			{
				delete NodeForwardStarArray[i].OutgoingLinkNoArray;
				delete NodeForwardStarArray[i].OutgoingNodeNoArray;
			}

		}



		if (NodeForwardStarArray)
			delete NodeForwardStarArray;
				
		if(temp_path_node_vector!=NULL)
		{ 
		delete temp_path_node_vector;
		delete temp_path_link_vector;
		}


		if (m_SENodeList != NULL)  //1
			delete m_SENodeList;

		if (m_node_status_array != NULL)  //2
			delete m_node_status_array;

		if (m_label_time_array != NULL)  //3
			delete m_label_time_array;

		if (m_label_distance_array != NULL)  //4
			delete m_label_distance_array;

		if (m_node_predecessor != NULL)  //5
			delete m_node_label_cost;

		if (m_link_predecessor != NULL)  //6
			delete m_link_predecessor;

		if (m_node_label_cost != NULL)  //7
			delete m_node_label_cost;

			   
		if (m_link_predecessor != NULL)
			delete m_link_predecessor;

		if (m_link_flow_volume_array != NULL)
			delete m_link_flow_volume_array;
	}


	// SEList: scan eligible List implementation: the reason for not using STL-like template is to avoid overhead associated pointer allocation/deallocation
	inline void SEList_clear()
	{
		m_ListFront = -1;
		m_ListTail = -1;
	}

	inline void SEList_push_front(int node)
	{
		if (m_ListFront == -1)  // start from empty
		{
			m_SENodeList[node] = -1;
			m_ListFront = node;
			m_ListTail = node;
		}
		else
		{
			m_SENodeList[node] = m_ListFront;
			m_ListFront = node;
		}
	}
	inline void SEList_push_back(int node)
	{
		if (m_ListFront == -1)  // start from empty
		{
			m_ListFront = node;
			m_ListTail = node;
			m_SENodeList[node] = -1;
		}
		else
		{
			m_SENodeList[m_ListTail] = node;
			m_SENodeList[node] = -1;
			m_ListTail = node;
		}
	}

	inline bool SEList_empty()
	{
		return(m_ListFront == -1);
	}

	inline int SEList_front()
	{
		return m_ListFront;
	}

	inline void SEList_pop_front()
	{
		int tempFront = m_ListFront;
		m_ListFront = m_SENodeList[m_ListFront];
		m_SENodeList[tempFront] = -1;
	}


	//major function: update the cost for each node at each SP tree, using a stack from the origin structure 

	void backtrace_shortest_path_tree(Assignment& assignment, int iteration_number, int o_node_index);
	void update_resource_consumption_before_SP_calculation(Assignment& assignment, int iteration_number_outterloop);

	//major function 2: // time-dependent label correcting algorithm with double queue implementation
	float optimal_label_correcting(Assignment& assignment, int iteration_k, int o_node_index, int d_node_no  = -1, bool pure_travel_time_cost = false)
	{	
		int SE_loop_count = 0;
		if (iteration_k == 0)
		{
			BuildNetwork(assignment);  // based on agent type and link type
		}

		int origin_node = m_origin_node_vector[o_node_index]; // assigned nodes for computing 
		int origin_zone = m_origin_zone_seq_no_vector[o_node_index]; // assigned nodes for computing 
		int agent_type = m_agent_type_no; // assigned nodes for computing 

		if (g_debugging_flag && assignment.g_pFileDebugLog != NULL)
			fprintf(assignment.g_pFileDebugLog, "SP iteration k = %d: origin node: %d, agent type %d \n", iteration_k, g_node_vector[origin_node].node_id, m_agent_type_no);

		for (int i = 0; i < assignment.g_number_of_nodes; i++) //Initialization for all non-origin nodes
		{
			m_node_status_array[i] = 0;  // not scanned
			m_node_label_cost[i] = _MAX_LABEL_COST;
			m_link_predecessor[i] = -1;  // pointer to previous NODE INDEX from the current label at current node and time
			m_node_predecessor[i] = -1;  // pointer to previous NODE INDEX from the current label at current node and time
			//m_node_label_cost_withouttoll[i] = _MAX_LABEL_COST;
			// comment out to speed up comuting 
			////m_label_time_array[i] = 0;
			////m_label_distance_array[i] = 0;
		}


		int internal_debug_flag = 0;
		if (NodeForwardStarArray[origin_node].OutgoingLinkSize == 0)
		{
			return 0;
		}

		//Initialization for origin node at the preferred departure time, at departure time, cost = 0, otherwise, the delay at origin node
		m_label_time_array[origin_node] = 0;
		m_node_label_cost[origin_node] = 0.0;
		m_label_distance_array[origin_node] = 0.0;

		//m_node_label_cost_withouttoll[origin_node] = 0.0;

		SEList_clear();
		SEList_push_back(origin_node);

		int i, from_node, to_node;
		int link_sqe_no;
		bool  b_node_updated;
		CLink* pLink;
		float new_time, new_distance, new_to_node_cost;
		while (!SEList_empty())
		{
			SE_loop_count++;

			from_node = SEList_front();//pop a node FromID for scanning

			SEList_pop_front();  // remove current node FromID from the SE list
			m_node_status_array[from_node] = 2;

			if (g_debugging_flag && assignment.g_pFileDebugLog !=NULL)
				fprintf(assignment.g_pFileDebugLog, "SP: SE node: %d\n", g_node_vector[from_node].node_id);

	
			//scan all outbound nodes of the current node
			for (i = 0; i < NodeForwardStarArray[from_node].OutgoingLinkSize; i++)  // for each link (i,j) belong A(i)
			{

				to_node = NodeForwardStarArray[from_node].OutgoingNodeNoArray[i];

				if (to_node == origin_node)
					continue;


				b_node_updated = false;

				link_sqe_no = NodeForwardStarArray[from_node].OutgoingLinkNoArray[i];
				//if (g_node_vector [ g_link_vector[link_sqe_no].from_node_seq_no].node_id == 1537 && g_node_vector[g_link_vector[link_sqe_no].to_node_seq_no].node_id == 16133)
				//{
				//	cout << "a" << endl;
				//}

				pLink = &(g_link_vector[link_sqe_no]);
				if (pLink->zone_seq_no_for_outgoing_connector >= 0 )
				{
					if(pLink->zone_seq_no_for_outgoing_connector != origin_zone)
					{ 
					continue;  // filter out for an outgoing connector with a centriod zone id different from the origin zone seq no
					}
				}

				new_time = m_label_time_array[from_node] + pLink->travel_time_per_period[tau];

				new_distance = m_label_distance_array[from_node] + pLink->length;

				new_to_node_cost = m_node_label_cost[from_node] + pLink->travel_time_per_period[tau] + pLink->cost /m_value_of_time * 60;  // *60 as 60 min per hour

				if (new_to_node_cost < - 1000)
				{
					int i_debug_ = 1;
				}

				if(pure_travel_time_cost == false)
				{
					if(assignment.assignment_mode == 2)  // system optimal mode
					{ 
						new_to_node_cost += pLink->travel_marginal_cost_per_period[tau][m_agent_type_no];
					}

					if (assignment.assignment_mode == 3)  // exterior panalty mode
					{
						new_to_node_cost += pLink->exterior_penalty_derivative_per_period[tau][m_agent_type_no];
				
					}
				}

				//g_link_vector[m_node_vector[from_node].m_outgoing_link_seq_no_vector[i].link_seq_no].TollMAP[agent_type]] / max(0.0001, assignment.g_AgentTypeVector [agent_type]);
								//float new_to_node_cost_withouttoll = m_node_label_cost_withouttoll[from_node] + m_link_cost_withouttoll_array[link_entering_time_interval][agent_type][m_node_vector[from_node].m_outgoing_link_seq_no_vector[i].link_seq_no];

								/*if (g_debugging_flag && g_pFileDebugLog != NULL)
								{
									fprintf(g_pFileDebugLog, "SP: checking from node %d, to node %d  cost = %d\n",
										g_node_vector[from_node].node_id,
										g_node_vector[to_node].node_id,
										new_to_node_cost, g_node_vector[from_node].m_outgoing_link_seq_no_vector[i].cost);
								}*/

				if (new_to_node_cost < m_node_label_cost[to_node]) // we only compare cost at the downstream node ToID at the new arrival time t
				{

					if (g_debugging_flag && assignment.g_pFileDebugLog != NULL)
					{
						fprintf(assignment.g_pFileDebugLog, "SP: updating node: %d current cost: %.2f, new cost %.2f\n",
							g_node_vector[to_node].node_id,
							m_node_label_cost[to_node], new_to_node_cost);
					}

					// update cost label and node/time predecessor
					m_label_time_array[to_node] = new_time;
					m_label_distance_array[to_node] = new_distance;
					m_node_label_cost[to_node] = new_to_node_cost;
					//m_node_label_cost_withouttoll[to_node] = new_to_node_cost_withouttoll;
					m_node_predecessor[to_node] = from_node;  // pointer to previous physical NODE INDEX from the current label at current node and time
					m_link_predecessor[to_node] = link_sqe_no;  // pointer to previous physical NODE INDEX from the current label at current node and time

					b_node_updated = true;

					if (g_debugging_flag && assignment.g_pFileDebugLog != NULL)
						fprintf(assignment.g_pFileDebugLog, "SP: add node %d into SE List\n",
							g_node_vector[to_node].node_id);

					//to_node is zone centroid and not origin_node,is to make sure no passing zones, only needed in network with connector
					/*if (m_node_vector[to_node].zone_id != -1 && to_node != origin_node)
					{
						m_node_status_array[to_node] = 1;
					}*/


					// dequeu updating rule for m_node_status_array
					if (m_node_status_array[to_node] == 0)
					{
						SEList_push_back(to_node);
						m_node_status_array[to_node] = 1;
					}
					if (m_node_status_array[to_node] == 2)
					{
						SEList_push_front(to_node);
						m_node_status_array[to_node] = 1;
					}

				}

			}

		}

		if (g_debugging_flag && assignment.g_pFileDebugLog != NULL)
			{ 
			fprintf(assignment.g_pFileDebugLog, "SPtree at iteration k = %d: origin node: %d, agent type %d \n", iteration_k, g_node_vector[origin_node].node_id, m_agent_type_no);

				for (int i = 0; i < assignment.g_number_of_nodes; i++) //Initialization for all non-origin nodes
				{
					int node_pred_no = m_node_predecessor[i];
					int node_pred_id = -1;
					if (node_pred_no >= 0)
						node_pred_id = g_node_vector[node_pred_no].node_id;

					fprintf(assignment.g_pFileDebugLog, "SP node: %d, label cost %.3f, time %.3f, node_pred_id %d \n",
						g_node_vector[i].node_id,
						m_node_label_cost[i],
						m_label_time_array[i],
						node_pred_id
						);

				}

		}

		if (d_node_no >= 1)
			return m_node_label_cost[d_node_no];
		else 
			return 0;  // one to all shortest pat
	}


};

std::vector<CNode> g_node_vector;
std::vector<CLink> g_link_vector;
std::vector<COZone> g_zone_vector;
std::vector<CAGBMAgent> g_agbmagent_vector;
std::vector<NetworkForSP*> g_NetworkForSP_vector;

NetworkForSP g_RoutingNetwork;


class VehicleScheduleNetworks {

public:
	int m_agent_type_no;
	int m_time_interval_size;  // 1440

	std::vector<CNode> m_node_vector;  // local copy of node vector, based on agent type and origin node
	std::map<int, float> g_passenger_link_profit;  // link with negative link

	void BuildNetwork(Assignment& assignment, int tau, int iteration)
	{



		for (int i = 0; i < assignment.g_number_of_nodes; i++) //Initialization for all non-origin nodes
		{
			CNode node;  // create a node object 

			node.node_id = g_node_vector[i].node_id;
			node.node_seq_no = g_node_vector[i].node_seq_no;

			for (int j = 0; j < g_node_vector[i].m_outgoing_link_seq_no_vector.size(); j++)
			{

				int link_seq_no = g_node_vector[i].m_outgoing_link_seq_no_vector[j];

				if (assignment.g_LinkTypeMap[g_link_vector[link_seq_no].link_type].AllowAgentType(assignment.g_AgentTypeVector[m_agent_type_no].agent_type))  // only predefined allowed agent type can be considered
				{
					int from_node_seq_no = g_link_vector[link_seq_no].from_node_seq_no;
					node.m_outgoing_link_seq_no_vector.push_back(link_seq_no);

					g_passenger_link_profit[link_seq_no] = g_link_vector[link_seq_no].get_generalized_first_order_gradient_cost_of_second_order_loss_for_agent_type(tau, m_agent_type_no);

					if (g_debugging_flag && assignment.g_pFileDebugLog != NULL)
						fprintf(assignment.g_pFileDebugLog, "DP iteration %d: link %d->%d:  profit %.3f\n", 
							iteration,
							g_node_vector[g_link_vector[link_seq_no].from_node_seq_no].node_id,
							g_node_vector[g_link_vector[link_seq_no].to_node_seq_no].node_id,
							g_passenger_link_profit[link_seq_no]);



					node.m_to_node_seq_no_vector.push_back(g_node_vector[i].m_to_node_seq_no_vector[j]);
				}

			}

			m_node_vector.push_back(node);

		}
	}


	//class for vehicle scheduling states
	class CVSState
	{
	public:
		int current_node_no;  // space dimension

		std::map<int, int> passenger_service_state;  // passenger means link with negative costs

		std::vector<int> m_visit_link_sequence;  // store link sequence

		int m_vehicle_capacity;

		float LabelCost;  // with LR price
		float LabelTime;   // sum of travel time up to now, arrival time at node


		CVSState()
		{
			current_node_no = 0;
			LabelTime = 0;
			LabelCost = 0;
			m_vehicle_capacity = 0;
		}

		void Copy(CVSState* pSource)
		{
			current_node_no = pSource->current_node_no;
			passenger_service_state.clear();
			passenger_service_state = pSource->passenger_service_state;


			m_visit_link_sequence = pSource->m_visit_link_sequence;
			m_vehicle_capacity = pSource->m_vehicle_capacity;
			LabelCost = pSource->LabelCost;
			LabelTime = pSource->LabelTime;
		}
		int GetPassengerLinkServiceState(int link_no)
		{
			if (passenger_service_state.find(link_no) != passenger_service_state.end())
				return passenger_service_state[link_no];  // 1 or 2
			else
				return 0;
		}



		std::string generate_string_key()
		{

			stringstream s;

			s << "n";
			s << current_node_no;  // space key
			for (std::map<int, int>::iterator it = passenger_service_state.begin(); it != passenger_service_state.end(); ++it)
			{
				s << "_";

				s << it->first << "[" << it->first << "]";

			}
			string converted(s.str());
			return converted;

		}

		bool operator<(const CVSState& other) const
		{
			return LabelCost < other.LabelCost;
		}

	};

	class C_time_indexed_state_vector
	{
	public:
		int current_time;


		std::vector<CVSState> m_VSStateVector;

		std::map<std::string, int> m_state_map;

		void Reset()
		{
			current_time = 0;
			m_VSStateVector.clear();
			m_state_map.clear();
		}

		int m_find_state_index(std::string string_key)
		{

			if (m_state_map.find(string_key) != m_state_map.end())
			{
				return m_state_map[string_key];
			}
			else
				return -1;  // not found

		}

		void update_state(CVSState new_element)
		{
			std::string string_key = new_element.generate_string_key();//if it is new, string is n100, no state index
			int state_index = m_find_state_index(string_key);

			if (state_index == -1)  // no such state at this time
			{
				// add new state
				state_index = m_VSStateVector.size();
				m_VSStateVector.push_back(new_element);
				m_state_map[string_key] = state_index;
			}
			else
			{//DP
				if (new_element.LabelCost < m_VSStateVector[state_index].LabelCost)
				{
					m_VSStateVector[state_index].Copy(&new_element);
				}

			}

		}

		void Sort()
		{
			std::sort(m_VSStateVector.begin(), m_VSStateVector.end());

			m_state_map.clear(); // invalid
		}

		void SortAndCleanEndingState(int BestKValue)
		{
			if (m_VSStateVector.size() > 2 * BestKValue)
			{
				std::sort(m_VSStateVector.begin(), m_VSStateVector.end());

				m_state_map.clear(); // invalid
				m_VSStateVector.erase(m_VSStateVector.begin() + BestKValue, m_VSStateVector.end());
			}
		}

		float GetBestValue()
		{
			// LabelCost not PrimalCost when sorting
			std::sort(m_VSStateVector.begin(), m_VSStateVector.end());

			if (m_VSStateVector.size() >= 1)
			{
				std::string state_str = m_VSStateVector[0].generate_string_key();

				return m_VSStateVector[0].LabelCost;

			}
			else
				return _MAX_LABEL_COST;
		}

		std::vector<int> GetBestLinkSequence()
		{
			std::vector <int> link_sequence;
			// LabelCost not PrimalCost when sorting
			std::sort(m_VSStateVector.begin(), m_VSStateVector.end());

			if (m_VSStateVector.size() >= 1)
			{
				std::string state_str = m_VSStateVector[0].generate_string_key();

				if(m_VSStateVector[0].m_visit_link_sequence.size() > 0 )
					return m_VSStateVector[0].m_visit_link_sequence;
				else
					return link_sequence;

			}

			return link_sequence;
		}

	};

	//vehicle state at time t

	// for collecting the final feasible states accesible to the depot
	C_time_indexed_state_vector g_ending_state_vector;

	C_time_indexed_state_vector** g_time_dependent_state_vector;  // label cost vector [i,t,w]


	void AllocateVSNMemory(int number_of_nodes)
	{
		g_time_dependent_state_vector = AllocateDynamicArray <C_time_indexed_state_vector>(number_of_nodes, m_time_interval_size);  //1
	}

	~VehicleScheduleNetworks()
	{
		DeallocateDynamicArray(g_time_dependent_state_vector, g_node_vector.size(), m_time_interval_size);
	}


	std::vector<int> g_optimal_time_dependenet_dynamic_programming(
		int origin_node,
		int destination_node,
		int vehicle_capacity,
		int demand_time_period_no,
		//maximum choose
		int BestKSize)
		// time-dependent label correcting algorithm with double queue implementation
	{
		int arrival_time = m_time_interval_size - 1;  // restrict the search range.

		int t = 0;
		//step 2: Initialization for origin node at the preferred departure time, at departure time
		for (int i = 0; i < assignment.g_number_of_nodes; i++)
		{
			g_time_dependent_state_vector[i][0].Reset();

		}
		g_ending_state_vector.Reset();

		CVSState element;

		element.current_node_no = origin_node;
		g_time_dependent_state_vector[origin_node][0].update_state(element);


		// step 3: //dynamic programming
		for (t = 0; t < arrival_time; t++)  //first loop: time
		{

			for (int n = 0; n < m_node_vector.size(); n++)
			{
				// step 1: sort m_VSStateVector by labelCost for scan best k elements in step2
				g_time_dependent_state_vector[n][t].Sort();

				// step 2: scan the best k elements
				for (int w_index = 0; w_index < min(BestKSize, g_time_dependent_state_vector[n][t].m_VSStateVector.size()); w_index++)
				{
					CVSState* pElement = &(g_time_dependent_state_vector[n][t].m_VSStateVector[w_index]);

					int from_node = pElement->current_node_no;

					// step 2.1 link from node to toNode
					for (int i = 0; i < m_node_vector[from_node].m_outgoing_link_seq_no_vector.size(); i++)
					{
						int link_seq_no = m_node_vector[from_node].m_outgoing_link_seq_no_vector[i];
						int to_node = g_link_vector[link_seq_no].to_node_seq_no;

						float new_time = pElement->LabelTime + g_link_vector[link_seq_no].travel_time_per_period[demand_time_period_no];
						int new_time_int = max(pElement->LabelTime + 1, (int)(new_time + 0.5));  // move at least one time step further

						// step 2.2. check feasibility of node type with the current element
						if (new_time <= arrival_time)
						{

							// skip scanning when the origin/destination nodes arrival time is out of time window
							//feasible state transitions
								// do not need waiting
							CVSState new_element;
							new_element.Copy(pElement);

							new_element.current_node_no = to_node;

							new_element.LabelCost += g_link_vector[link_seq_no].travel_time_per_period[demand_time_period_no] / 60.0 * assignment.g_AgentTypeVector[m_agent_type_no].value_of_time; // 60.0 is to convert hour to 60 min as VOT is denoted as dollars per hour
							if (g_passenger_link_profit.find(link_seq_no) != g_passenger_link_profit.end())
							{
								new_element.LabelCost += g_passenger_link_profit[link_seq_no];// + negative cost
								new_element.passenger_service_state[link_seq_no] = 1;  // mark carry status
								new_element.m_vehicle_capacity -= 1;

							}


							new_element.m_visit_link_sequence.push_back(link_seq_no);
							g_time_dependent_state_vector[to_node][new_time_int].update_state(new_element);

							if (to_node == destination_node)
							{

								//time window of destination_node
								if (new_time < arrival_time)
								{
									g_ending_state_vector.update_state(new_element);
									g_ending_state_vector.SortAndCleanEndingState(BestKSize);
								}
							}
						}

					}
				}
			}  // for all nodes
		} // for all time t


	// no backf
		return g_ending_state_vector.GetBestLinkSequence();

	}

};
//    This file is part of FlashDTA.

//    FlashDTA is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.

//    FlashDTA  is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.

//    You should have received a copy of the GNU General Public License
//    along with DTALite.  If not, see <http://www.gnu.org/licenses/>.


int g_read_integer(FILE* f, bool speicial_char_handling)
// read an integer from the current pointer of the file, skip all spaces
{
	char ch, buf[32];
	int i = 0;
	int flag = 1;
	/* returns -1 if end of file is reached */

	while (true)
	{
		ch = getc(f);
		//cout << "get from node successful: " << ch;
		if (ch == EOF || (speicial_char_handling && (ch == '*' || ch == '$')))
			return -1; // * and $ are special characters for comments
		if (isdigit(ch))
			break;
		if (ch == '-')
			flag = -1;
		else
			flag = 1;
	};
	if (ch == EOF) return -1;


	while (isdigit(ch)) {
		buf[i++] = ch;
		//cout << "isdigit" << buf[i++] << endl;
		ch = fgetc(f);
		//cout << "new ch" << ch;
	}
	buf[i] = 0;


	return atoi(buf) * flag;

}


float g_read_float(FILE *f)
//read a floating point number from the current pointer of the file,
//skip all spaces

{
	char ch, buf[32];
	int i = 0;
	int flag = 1;

	/* returns -1 if end of file is reached */

	while (true)
	{
		ch = getc(f);
		if (ch == EOF || ch == '*' || ch == '$') return -1;
		if (isdigit(ch))
			break;

		if (ch == '-')
			flag = -1;
		else
			flag = 1;

	};
	if (ch == EOF) return -1;
	while (isdigit(ch) || ch == '.') {
		buf[i++] = ch;
		ch = fgetc(f);

	}
	buf[i] = 0;

	/* atof function converts a character string (char *) into a doubleing
	pointer equivalent, and if the string is not a floting point number,
	a zero will be return.
	*/

	return (float)(atof(buf) * flag);

}



//split the string by "_"
vector<string> split(const string &s, const string &seperator) {
	vector<string> result;
	typedef string::size_type string_size;
	string_size i = 0;

	while (i != s.size()) {
		int flag = 0;
		while (i != s.size() && flag == 0) {
			flag = 1;
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[i] == seperator[x]) {
					++i;
					flag = 0;
					break;
				}
		}

		flag = 0;
		string_size j = i;
		while (j != s.size() && flag == 0) {
			for (string_size x = 0; x < seperator.size(); ++x)
				if (s[j] == seperator[x]) {
					flag = 1;
					break;
				}
			if (flag == 0)
				++j;
		}
		if (i != j) {
			result.push_back(s.substr(i, j - i));
			i = j;
		}
	}
	return result;
}

vector<float> g_time_parser(vector<string>& inputstring)
{
	vector<float> output_global_minute;

	for (int k = 0; k < inputstring.size(); k++)
	{
		vector<string> sub_string = split(inputstring[k], "_");

		for (int i = 0; i < sub_string.size(); i++)
		{
			//HHMM
			//012345
			char hh1 = sub_string[i].at(0);
			char hh2 = sub_string[i].at(1);
			char mm1 = sub_string[i].at(2);
			char mm2 = sub_string[i].at(3);

			float hhf1 = ((float)hh1 - 48);
			float hhf2 = ((float)hh2 - 48);
			float mmf1 = ((float)mm1 - 48);
			float mmf2 = ((float)mm2 - 48);

			float hh = hhf1 * 10 * 60 + hhf2 * 60;
			float mm = mmf1 * 10 + mmf2;
			float global_mm_temp = hh + mm;
			output_global_minute.push_back(global_mm_temp);
		}
	}

	return output_global_minute;
} // transform hhmm to minutes 


inline string g_time_coding(float time_stamp)
{
	int hour = time_stamp / 60;
	int minute = time_stamp - hour * 60;

	ostringstream strm;
	strm.fill('0');
	strm << setw(2) << hour << setw(2) << minute;

	return strm.str();
} // transform hhmm to minutes 


void g_ProgramStop()
{

	cout << "STALite Program stops. Press any key to terminate. Thanks!" << endl;
	getchar();
	exit(0);
};



//void ReadLinkTollScenarioFile(Assignment& assignment)
//{
//
//	for (unsigned li = 0; li < g_link_vector.size(); li++)
//	{
//
//		g_link_vector[li].TollMAP.erase(g_link_vector[li].TollMAP.begin(), g_link_vector[li].TollMAP.end()); // remove all previouly read records
//	}
//
//	// generate toll based on demand type code in input_link.csv file
//	int demand_flow_type_count = 0;
//
//	for (unsigned li = 0; li < g_link_vector.size(); li++)
//	{
//		if (g_link_vector[li].agent_type_code.size() >= 1)
//		{  // with data string
//
//			std::string agent_type_code = g_link_vector[li].agent_type_code;
//
//			vector<float> TollRate;
//			for (int at = 0; at < assignment.g_AgentTypeVector.size(); at++)
//			{
//				CString number;
//				number.Format(_T("%d"), at);
//
//				std::string str_number = CString2StdString(number);
//				if (agent_type_code.find(str_number) == std::string::npos)   // do not find this number
//				{
//					g_link_vector[li].TollMAP[at] = 999;
//					demand_flow_type_count++;
//				}
//				else
//				{
//					g_link_vector[li].TollMAP[at] = 0;
//				}
//
//			}  //end of pt
//		}
//	}
//}



int g_ParserIntSequence(std::string str, std::vector<int>& vect)
{

	std::stringstream ss(str);

	int i;

	while (ss >> i)
	{
		vect.push_back(i);

		if (ss.peek() == ';')
			ss.ignore();
	}

	return vect.size();
}

void g_ReadDemandFileBasedOnDemandFileList(Assignment& assignment)
{


//	fprintf(g_pFileOutputLog, "number of zones =,%lu\n", g_zone_vector.size());

	assignment.InitializeDemandMatrix(g_zone_vector.size(), assignment.g_AgentTypeVector.size(), assignment.g_DemandPeriodVector.size());

	float total_demand_in_demand_file = 0;

	CCSVParser parser;
	cout << "Step 4: Reading file demand_file_list.csv..." << endl;

	if (parser.OpenCSVFile("demand_file_list.csv", true))
	{
		int i = 0;

		while (parser.ReadRecord())
		{

			int file_sequence_no = 1;
			string file_name;
			string format_type = "null";

			string demand_period, agent_type;

			int demand_format_flag = 0;

			if (parser.GetValueByFieldName("file_sequence_no", file_sequence_no) == false)
				break;

			if (file_sequence_no <= -1)  // skip negative sequence no 
				continue;

			parser.GetValueByFieldName("file_name", file_name);

			parser.GetValueByFieldName("demand_period", demand_period);


			parser.GetValueByFieldName("format_type", format_type);
			if (format_type.find("null") != string::npos)  // skip negative sequence no 
			{
				cout << "Please provide format_type in file demand_file_list.csv" << endl;
				g_ProgramStop();
			}


			double total_ratio = 0;

			parser.GetValueByFieldName("agent_type", agent_type);


			int agent_type_no = 0;
			int demand_period_no = 0;

			if (assignment.demand_period_to_seqno_mapping.find(demand_period) != assignment.demand_period_to_seqno_mapping.end())
			{
				demand_period_no = assignment.demand_period_to_seqno_mapping[demand_period];

			}
			else
			{
				cout << "Error: demand period in demand_file_list " << demand_period << "cannot be found." << endl;
				g_ProgramStop();

			}

			bool b_multi_agent_list = false;

			if (agent_type == "multi_agent_list")
			{
				b_multi_agent_list = true;
			}else
			{ 

				if (assignment.agent_type_2_seqno_mapping.find(agent_type) != assignment.agent_type_2_seqno_mapping.end())
				{
					agent_type_no = assignment.agent_type_2_seqno_mapping[agent_type];

				}
				else
				{
					cout << "Error: agent_type in agent_type " << agent_type << "cannot be found." << endl;
					g_ProgramStop();
				}
			}

			if (demand_period_no > _MAX_TIMEPERIODS)
			{
				cout << "demand_period_no should be less than settings in demand_period.csv. Please change the parameter settings in the source code." << endl;
				g_ProgramStop();
			}

			if (format_type.find("column") != string::npos)  // or muliti-column
			{


				bool bFileReady = false;
				
				FILE* st;
				// read the file formaly after the test. 

				int error_count = 0;
				fopen_ss(&st, file_name.c_str(), "r");
				if (st != NULL)
				{

					bFileReady = true;
					int line_no = 0;

					while (true)
					{
						int origin_zone = g_read_integer(st, true);
						int destination_zone = g_read_integer(st, true);
						float demand_value = g_read_float(st);

						if (origin_zone <= 0)
						{

							if (line_no == 1 && !feof(st))  // read only one line, but has not reached the end of the line
							{
								cout << endl << "Error: Only one line has been read from file. Are there multiple columns of demand type in file " << file_name << " per line?" << endl;
								g_ProgramStop();

							}
							break;
						}

						if (assignment.g_zoneid_to_zone_seq_no_mapping.find(origin_zone) == assignment.g_zoneid_to_zone_seq_no_mapping.end())
						{
							if(error_count < 10)
								cout << endl << "Warning: origin zone " << origin_zone << "  has not been defined in node.csv" << endl;

							error_count++;
							continue; // origin zone  has not been defined, skipped. 
						}



						if (assignment.g_zoneid_to_zone_seq_no_mapping.find(destination_zone) == assignment.g_zoneid_to_zone_seq_no_mapping.end())
						{
							if (error_count < 10)
								cout << endl << "Warning: destination zone " << destination_zone << "  has not been defined in node.csv" << endl;

							error_count++;
							continue; // destination zone  has not been defined, skipped. 
						}

						int from_zone_seq_no = 0;
						int to_zone_seq_no = 0;
						from_zone_seq_no = assignment.g_zoneid_to_zone_seq_no_mapping[origin_zone];
						to_zone_seq_no = assignment.g_zoneid_to_zone_seq_no_mapping[destination_zone];

						if (assignment.g_zoneid_to_zone_seq_no_mapping.size() >= 395)
						{
							int i_debug = 1;
						}
							if (demand_value < -99) // encounter return 
							{
								break;
							}

							if (from_zone_seq_no == 0 && to_zone_seq_no == 3)
							{
								int ibebug = 1;
							}

							assignment.total_demand[agent_type_no][demand_period_no] += demand_value;
							assignment.g_column_pool[from_zone_seq_no][to_zone_seq_no][agent_type_no][demand_period_no].od_volume += demand_value;
							assignment.total_demand_volume += demand_value;
							assignment.g_origin_demand_array[from_zone_seq_no][agent_type_no][demand_period_no] += demand_value;

							// we generate vehicles here for each OD data line
							if (line_no <= 5)  // read only one line, but has not reached the end of the line
								cout << "o_zone_id:" << origin_zone << ", d_zone_id: " << destination_zone << ", value = " << demand_value << endl;


						line_no++;
					}  // scan lines


					fclose(st);

					cout << "total_demand_volume is " << assignment.total_demand_volume << endl;
				}
				else  //open file
				{
					cout << "Error: File " << file_name << " cannot be opened.\n It might be currently used and locked by EXCEL." << endl;
					g_ProgramStop();

				}
			}

			else if (format_type.compare("agent_csv") == 0)
			{

				CCSVParser parser;

				if (parser.OpenCSVFile(file_name, false))
				{
					int total_demand_in_demand_file = 0;


					// read agent file line by line,

					int agent_id, o_zone_id, d_zone_id;
					string agent_type, demand_period;
					
					std::vector <int> node_sequence;

					while (parser.ReadRecord())
					{
						total_demand_in_demand_file++;

						if (total_demand_in_demand_file % 1000 == 0)
							cout << "demand_volume is " << total_demand_in_demand_file << endl;

						parser.GetValueByFieldName("agent_id", agent_id);

						parser.GetValueByFieldName("o_zone_id", o_zone_id);
						parser.GetValueByFieldName("d_zone_id", d_zone_id);

						CAgentPath agent_path_element;

						int o_node_id;
						int d_node_id;

						parser.GetValueByFieldName("path_id", agent_path_element.path_id);
						parser.GetValueByFieldName("o_node_id", o_node_id);
						parser.GetValueByFieldName("d_node_id", d_node_id);
						parser.GetValueByFieldName("volume", agent_path_element.volume);
					
						agent_path_element.o_node_no = assignment.g_internal_node_to_seq_no_map[o_node_id];
						agent_path_element.d_node_no = assignment.g_internal_node_to_seq_no_map[d_node_id];


						int from_zone_seq_no = 0;
						int to_zone_seq_no = 0;
						from_zone_seq_no = assignment.g_zoneid_to_zone_seq_no_mapping[o_zone_id];
						to_zone_seq_no = assignment.g_zoneid_to_zone_seq_no_mapping[d_zone_id];


						assignment.total_demand[agent_type_no][demand_period_no] += agent_path_element.volume;
						assignment.g_column_pool[from_zone_seq_no][to_zone_seq_no][agent_type_no][demand_period_no].od_volume += agent_path_element.volume;
						assignment.total_demand_volume += agent_path_element.volume;
						assignment.g_origin_demand_array[from_zone_seq_no][agent_type_no][demand_period_no] += agent_path_element.volume;


						if (assignment.g_AgentTypeVector[agent_type_no].flow_type == 1)  // fixed path
						{
							bool bValid = true;


							std::string path_node_sequence;
							parser.GetValueByFieldName("node_sequence", path_node_sequence);

							std::vector<int> node_id_sequence;

							g_ParserIntSequence(path_node_sequence, node_id_sequence);

							std::vector<int> node_no_sequence;
							std::vector<int> link_no_sequence;


							int node_sum = 0;
							for (int i = 0; i < node_id_sequence.size(); i++)
							{

								if (assignment.g_internal_node_to_seq_no_map.find(node_id_sequence[i]) == assignment.g_internal_node_to_seq_no_map.end())
								{
									bValid = false;
									continue; //has not been defined

									// warning
								}

								int internal_node_seq_no = assignment.g_internal_node_to_seq_no_map[node_id_sequence[i]];  // map external node number to internal node seq no. 
								node_no_sequence.push_back(internal_node_seq_no);

								node_sum += internal_node_seq_no;
								if (i >= 1)
								{ // check if a link exists

									int link_seq_no = -1;
									int prev_node_seq_no = assignment.g_internal_node_to_seq_no_map[node_id_sequence[i - 1]];  // map external node number to internal node seq no. 

									int current_node_no = node_no_sequence[i];
									if (g_node_vector[prev_node_seq_no].m_to_node_seq_no_map.find(current_node_no) != g_node_vector[prev_node_seq_no].m_to_node_seq_no_map.end())
									{
										link_seq_no = g_node_vector[prev_node_seq_no].m_to_node_seq_no_map[node_no_sequence[i]];

										link_no_sequence.push_back(link_seq_no);
									}
									else
									{
										bValid = false;
									}


								}

							}


							if (bValid == true)
							{
								agent_path_element.node_sum = node_sum; // pointer to the node sum based path node sequence;
								agent_path_element.path_link_sequence = link_no_sequence;
							}

						}
						
						assignment.g_column_pool[from_zone_seq_no][to_zone_seq_no][agent_type_no][demand_period_no].discrete_agent_path_vector.push_back(agent_path_element);


						
					}
					


				}
				else  //open file
				{
					cout << "Error: File " << file_name << " cannot be opened.\n It might be currently used and locked by EXCEL." << endl;
					g_ProgramStop();

				}

			}

			else
			{
				cout << "Error: format_type = " << format_type << " is not supported. Currently STALite supports format such as column and agent_csv." << endl;
				g_ProgramStop();
			}
		}

	}

}



void g_ReadInputData(Assignment& assignment)
{

	//step 0:read demand period file
	CCSVParser parser_demand_period;
	cout << "Step 1: Reading file demand_period.csv..." << endl;
	//g_LogFile << "Step 7.1: Reading file input_agent_type.csv..." << g_GetUsedMemoryDataInMB() << endl;
	if (!parser_demand_period.OpenCSVFile("demand_period.csv", true))
	{
		cout << "demand_period.csv cannot be opened. " << endl;
		g_ProgramStop();

	}

	if (parser_demand_period.inFile.is_open() || parser_demand_period.OpenCSVFile("demand_period.csv", true))
	{

		while (parser_demand_period.ReadRecord())
		{

			CDemand_Period demand_period;

			if (parser_demand_period.GetValueByFieldName("demand_period_id", demand_period.demand_period_id) == false)
			{
				cout << "Error: Field demand_period_id in file demand_period cannot be read." << endl;
				g_ProgramStop();
				break;
			}

			if (parser_demand_period.GetValueByFieldName("demand_period", demand_period.demand_period) == false)
			{
				cout << "Error: Field demand_period in file demand_period cannot be read." << endl;
				g_ProgramStop();
				break;
			}
			


			vector<float> global_minute_vector;

			if (parser_demand_period.GetValueByFieldName("time_period", demand_period.time_period) == false)
			{ 
				cout << "Error: Field time_period in file demand_period cannot be read." << endl;
				g_ProgramStop();
				break;
			}

			vector<string> input_string;
			input_string.push_back(demand_period.time_period);
			//input_string includes the start and end time of a time period with hhmm format
			global_minute_vector = g_time_parser(input_string); //global_minute_vector incldue the starting and ending time
			if (global_minute_vector.size() == 2)
			{

				demand_period.starting_time_slot_no = global_minute_vector[0] / MIN_PER_TIMESLOT;
				demand_period.ending_time_slot_no = global_minute_vector[1] / MIN_PER_TIMESLOT;

				//cout << global_minute_vector[0] << endl;
				//cout << global_minute_vector[1] << endl;
			}

			assignment.demand_period_to_seqno_mapping[demand_period.demand_period] = assignment.g_DemandPeriodVector.size();

			assignment.g_DemandPeriodVector.push_back(demand_period);


		}
		parser_demand_period.CloseCSVFile();

		if(assignment.g_DemandPeriodVector.size() == 0)
		{
		cout << "Error:  File demand_period.csv has no information." << endl;
		g_ProgramStop();
		}

	}
	else
	{
		cout << "Error: File demand_period.csv cannot be opened.\n It might be currently used and locked by EXCEL." << endl;
		g_ProgramStop();
	}


	assignment.g_number_of_demand_periods = assignment.g_DemandPeriodVector.size();
	//step 1:read demand type file

	cout << "Reading file link_type.csv..." << endl;

	CCSVParser parser_link_type;

	if (parser_link_type.OpenCSVFile("link_type.csv", true))
	{

		int line_no = 0;

		while (parser_link_type.ReadRecord())
		{
			CLinkType element;

			if (parser_link_type.GetValueByFieldName("link_type", element.link_type) == false)
			{
				if (line_no == 0)
				{
					cout << "Error: Field link_type cannot be found in file link_type.csv." << endl;
					g_ProgramStop();
				}
				else
				{  // read empty line
					break;
				}
			}

			if (assignment.g_LinkTypeMap.find(element.link_type) != assignment.g_LinkTypeMap.end())
			{
				cout << "Error: Field link_type " << element.link_type << " has been defined more than once in file link_type.csv." << endl;
				g_ProgramStop();

				break;
			}

			parser_link_type.GetValueByFieldName("type_code", element.type_code);
			parser_link_type.GetValueByFieldName("agent_type_list", element.agent_type_list);

			

			assignment.g_LinkTypeMap[element.link_type] = element;

			line_no++;
		}
	}
	else
	{
		cout << "Error: File link_type.csv cannot be opened.\n It might be currently used and locked by EXCEL." << endl;


	}


	CCSVParser parser_agent_type;
	cout << "Step 2: Reading file agent_type.csv..." << endl;
	//g_LogFile << "Step 7.1: Reading file input_agent_type.csv..." << g_GetUsedMemoryDataInMB() << endl;
	if (!parser_agent_type.OpenCSVFile("agent_type.csv", true))
	{
		cout << "agent_type.csv cannot be opened. " << endl;
		g_ProgramStop();

	}

	if (parser_agent_type.inFile.is_open() || parser_agent_type.OpenCSVFile("agent_type.csv", true))
	{
		assignment.g_AgentTypeVector.clear();
		while (parser_agent_type.ReadRecord())
		{

			CAgent_type agent_type;
			agent_type.agent_type_no = assignment.g_AgentTypeVector.size();

			if (parser_agent_type.GetValueByFieldName("agent_type", agent_type.agent_type) == false)
			{
				break;
			}

			parser_agent_type.GetValueByFieldName("VOT", agent_type.value_of_time);

			parser_agent_type.GetValueByFieldName("flow_type", agent_type.flow_type);
			// 0: flow, 1: fixed path, 2: integer decision variables.

			float value;
			

			std::map<int, CLinkType>::iterator it;

			// scan through the map with different node sum for different paths
			for (it = assignment.g_LinkTypeMap.begin();
				it != assignment.g_LinkTypeMap.end(); it++)
			{

				char field_name[20];

				sprintf_s(field_name, "PCE_link_type%d", it->first);
				if(parser_agent_type.GetValueByFieldName(field_name, value,false) == false)
				{
					cout << "Error: Field " << field_name << " in File agent_type.csv has not been defined. Please check\n " << endl;
					g_ProgramStop();
				}

				agent_type.PCE_link_type_map[it->first] = value;

				sprintf_s(field_name, "CRU_link_type%d", it->first);
				if (parser_agent_type.GetValueByFieldName(field_name, value, false) == false)
				{
					cout << "Error: Field " << field_name << " in File agent_type.csv has not been defined. Please check\n " << endl;
					g_ProgramStop();
				}

				agent_type.CRU_link_type_map[it->first] = value;



			}
	

			assignment.agent_type_2_seqno_mapping[agent_type.agent_type] = assignment.g_AgentTypeVector.size();



			assignment.g_AgentTypeVector.push_back(agent_type);
			assignment.g_number_of_agent_types = assignment.g_AgentTypeVector.size();

		}
		parser_agent_type.CloseCSVFile();

		if (assignment.g_AgentTypeVector.size() == 0 )
		{
			cout << "Error: File agent_type.csv does not contain information." << endl;
		}

	}
	else
	{
		cout << "Error: File agent_type.csv cannot be opened.\n It might be currently used and locked by EXCEL." << endl;
		g_ProgramStop();
	}


	if (assignment.g_AgentTypeVector.size() >= _MAX_AGNETTYPES)
	{
		cout << "Error: agent_type = " << assignment.g_AgentTypeVector.size() << " in file agent_type.csv is too large. " << "_MAX_AGNETTYPES = " << _MAX_AGNETTYPES << "Please contact program developers!";

		g_ProgramStop();
	}



	assignment.g_number_of_nodes = 0;
	assignment.g_number_of_links = 0;  // initialize  the counter to 0



	int internal_node_seq_no = 0;
	// step 3: read node file 

	std::map<int, int> zone_id_to_node_id_mapping;

	CCSVParser parser;
	if (parser.OpenCSVFile("node.csv", true))
	{



		while (parser.ReadRecord())  // if this line contains [] mark, then we will also read field headers.
		{

			int node_id;

			if (parser.GetValueByFieldName("node_id", node_id) == false)
				continue;

			if (assignment.g_internal_node_to_seq_no_map.find(node_id) != assignment.g_internal_node_to_seq_no_map.end())
			{
				continue; //has been defined
			}
			assignment.g_internal_node_to_seq_no_map[node_id] = internal_node_seq_no;


			CNode node;  // create a node object 

			node.node_id = node_id;
			node.node_seq_no = internal_node_seq_no;

			int zone_id; 
			parser.GetValueByFieldName("zone_id", zone_id);


			if(zone_id>=1)
			{ 
				if (zone_id_to_node_id_mapping.find(zone_id) == zone_id_to_node_id_mapping.end())
				{
					zone_id_to_node_id_mapping[zone_id] = node_id;
					node.zone_id = zone_id;
				}
				else
				{
					cout << "warning: zone_id " << zone_id << " have been defined more than once." << endl;

				}

			}

			/*node.x = x;
			node.y = y;*/
			internal_node_seq_no++;

			g_node_vector.push_back(node);  // push it to the global node vector

			assignment.g_number_of_nodes++;
			if (assignment.g_number_of_nodes % 1000 == 0)
				cout << "reading " << assignment.g_number_of_nodes << " nodes.. " << endl;
		}

		cout << "number of nodes = " << assignment.g_number_of_nodes << endl;

	//	fprintf(g_pFileOutputLog, "number of nodes =,%d\n", assignment.g_number_of_nodes);


		parser.CloseCSVFile();
	}

	// initialize zone vector
	for (int i = 0; i < g_node_vector.size(); i++)
	{


		if (g_node_vector[i].zone_id >= 1 
			&& zone_id_to_node_id_mapping.find(g_node_vector[i].zone_id) != zone_id_to_node_id_mapping.end() /* uniquely defined*/
			&& assignment.g_zoneid_to_zone_seq_no_mapping.find(g_node_vector[i].zone_id) == assignment.g_zoneid_to_zone_seq_no_mapping.end())  // create a new zone  // we assume each zone only has one node
		{ // we need to make sure we only create a zone in the memory if only there is positive demand flow from the (new) OD table
			COZone ozone;
			ozone.node_seq_no = g_node_vector[i].node_seq_no;
			ozone.zone_id = g_node_vector[i].zone_id;
			ozone.zone_seq_no = g_zone_vector.size();
			assignment.g_zoneid_to_zone_seq_no_mapping[ozone.zone_id] = assignment.g_zoneid_to_zone_seq_no_mapping.size();  // create the zone id to zone seq no mapping

			g_zone_vector.push_back(ozone);  // add element into vector
											 //	cout << ozone.zone_id << ' ' << ozone.zone_seq_no << endl;
		}
	}

	cout << "number of zones = " << g_zone_vector.size() << endl;
	// step 4: read link file 

	CCSVParser parser_link;

	if (parser_link.OpenCSVFile("road_link.csv", true))
	{
		while (parser_link.ReadRecord())  // if this line contains [] mark, then we will also read field headers.
		{
			int from_node_id;
			int to_node_id;
			if (parser_link.GetValueByFieldName("from_node_id", from_node_id) == false)
				continue;
			if (parser_link.GetValueByFieldName("to_node_id", to_node_id) == false)
				continue;

			int linkID = 0;
			parser_link.GetValueByFieldName("road_link_id", linkID);


			// add the to node id into the outbound (adjacent) node list

			int internal_from_node_seq_no = assignment.g_internal_node_to_seq_no_map[from_node_id];  // map external node number to internal node seq no. 
			int internal_to_node_seq_no = assignment.g_internal_node_to_seq_no_map[to_node_id];

			CLink link;  // create a link object 

			link.from_node_seq_no = internal_from_node_seq_no;
			link.to_node_seq_no = internal_to_node_seq_no;
			link.link_seq_no = assignment.g_number_of_links;
			link.to_node_seq_no = internal_to_node_seq_no;
			link.link_id = linkID;

			parser_link.GetValueByFieldName("facility_type", link.type);
			parser_link.GetValueByFieldName("link_type", link.link_type);

			if (assignment.g_LinkTypeMap.find(link.link_type) == assignment.g_LinkTypeMap.end())
			{
				cout << "link type " << link.link_type << " in road_link.csv is not defined in link_type.csv" <<endl;
				g_ProgramStop();

			}

			for (int at = 0; at < assignment.g_AgentTypeVector.size(); at++)
			{
				float PCE_ratio = assignment.g_AgentTypeVector[at].PCE_link_type_map[link.link_type];
			
				link.PCE_at[at] = PCE_ratio;


				float CRU = assignment.g_AgentTypeVector[at].CRU_link_type_map[link.link_type];

				link.CRU_at[at] = CRU;

			}




			if (assignment.g_LinkTypeMap[link.link_type].type_code == "c" && g_node_vector[internal_from_node_seq_no].zone_id >=0)
			{
				if(assignment.g_zoneid_to_zone_seq_no_mapping.find(g_node_vector[internal_from_node_seq_no].zone_id) != assignment.g_zoneid_to_zone_seq_no_mapping.end())
				link.zone_seq_no_for_outgoing_connector = assignment.g_zoneid_to_zone_seq_no_mapping [g_node_vector[internal_from_node_seq_no].zone_id];
			}


			parser_link.GetValueByFieldName("cost", link.cost);

				float length = 1.0; // km or mile
			float free_speed = 1.0;
			float k_jam = 200;
			parser_link.GetValueByFieldName("length", length);
			parser_link.GetValueByFieldName("free_speed", free_speed);

			int number_of_lanes = 1;
			parser_link.GetValueByFieldName("lanes", number_of_lanes);

			char VDF_field_name[20];

			for (int tau = 0; tau < assignment.g_number_of_demand_periods; tau++)
			{
				int demand_period_id = assignment.g_DemandPeriodVector[tau].demand_period_id;
				sprintf_s (VDF_field_name, "VDF_fftt%d", demand_period_id);
				parser_link.GetValueByFieldName(VDF_field_name, link.VDF_period[tau].FFTT);

				sprintf_s (VDF_field_name, "VDF_cap%d", demand_period_id);
				parser_link.GetValueByFieldName(VDF_field_name, link.VDF_period[tau].capacity);

				sprintf_s (VDF_field_name, "VDF_alpha%d", demand_period_id);
				parser_link.GetValueByFieldName(VDF_field_name, link.VDF_period[tau].alpha);

				sprintf_s (VDF_field_name, "VDF_beta%d", demand_period_id);
				parser_link.GetValueByFieldName(VDF_field_name, link.VDF_period[tau].beta);

				sprintf_s(VDF_field_name, "VDF_theta%d", demand_period_id);
				parser_link.GetValueByFieldName(VDF_field_name, link.VDF_period[tau].theta);

				//sprintf_s (VDF_field_name, "VDF_mu%d", demand_period_id);
				//parser_link.GetValueByFieldName(VDF_field_name, link.VDF_period[tau].mu);

				//sprintf_s (VDF_field_name, "VDF_gamma%d", demand_period_id);
				//parser_link.GetValueByFieldName(VDF_field_name, link.VDF_period[tau].gamma);

				if(assignment.assignment_mode == 3)
				{ 

				sprintf_s(VDF_field_name, "RUC_rho%d", demand_period_id);
				parser_link.GetValueByFieldName(VDF_field_name, link.VDF_period[tau].rho);

				sprintf_s(VDF_field_name, "RUC_resource%d", demand_period_id);
				parser_link.GetValueByFieldName(VDF_field_name, link.VDF_period[tau].ruc_base_resource,false);

				link.VDF_period[tau].starting_time_slot_no = assignment.g_DemandPeriodVector[tau].starting_time_slot_no;
				link.VDF_period[tau].ending_time_slot_no = assignment.g_DemandPeriodVector[tau].ending_time_slot_no;
				link.VDF_period[tau].period = assignment.g_DemandPeriodVector[tau].time_period;

				}

			}

			if (assignment.assignment_mode == 3)
			{
				parser_link.GetValueByFieldName("RUC_type", link.RUC_type);
			}

			// for each period

			float default_cap = 1000;
			float default_BaseTT = 1;

			// setup default value
			for (int tau = 0; tau < assignment.g_number_of_demand_periods; tau++)
			{
				link.TDBaseTT[tau] = default_BaseTT;
				link.TDBaseCap[tau] = default_cap;
			}

			//link.m_OutflowNumLanes = number_of_lanes;//visum lane_cap is actually link_cap

			link.link_spatial_capacity = k_jam * number_of_lanes*length;

			link.free_flow_travel_time_in_min = default_BaseTT;

			link.length = length;
			for (int tau = 0; tau < assignment.g_number_of_demand_periods; tau++)
			{
				link.travel_time_per_period[tau] = length / free_speed * 60;
			}
			// min // calculate link cost based length and speed limit // later we should also read link_capacity, calculate delay 

			//int sequential_copying = 0;
			//
			//parser_link.GetValueByFieldName("sequential_copying", sequential_copying);

			g_node_vector[internal_from_node_seq_no].m_outgoing_link_seq_no_vector.push_back(link.link_seq_no);  // add this link to the corresponding node as part of outgoing node/link
			g_node_vector[internal_from_node_seq_no].m_to_node_seq_no_vector .push_back(link.to_node_seq_no);  // add this link to the corresponding node as part of outgoing node/link
			g_node_vector[internal_from_node_seq_no].m_to_node_seq_no_map[link.to_node_seq_no] = link.link_seq_no;  // add this link to the corresponding node as part of outgoing node/link

			g_link_vector.push_back(link);

			assignment.g_number_of_links++;

			if (assignment.g_number_of_links % 1000 == 0)
				cout << "reading " << assignment.g_number_of_links << " links.. " << endl;
		}

		parser_link.CloseCSVFile();
	}
	// we now know the number of links
	cout << "number of links = " << assignment.g_number_of_links << endl;

//	fprintf(g_pFileOutputLog, "number of links =,%d\n", assignment.g_number_of_links);


};

void g_reset_link_volume_without_columns(int number_of_links, int iteration_index, bool b_self_reducing_path_volume)
{
	int number_of_demand_periods = assignment.g_number_of_demand_periods;
	int l, tau;
	if(iteration_index == 0)
	{ 

		for (l = 0; l < number_of_links; l++)
		{
			for (tau = 0; tau < number_of_demand_periods; tau++)
			{
				g_link_vector[l].flow_volume_per_period[tau] = 0; // used in travel time calculation
			}
		}
	}
	else
	{
		for (l = 0; l < number_of_links; l++)
		{
			for (tau = 0; tau < number_of_demand_periods; tau++)
			{
				if (b_self_reducing_path_volume == true)
				{							//after link volumn "tally", self-deducting the path volume by 1/(k+1) (i.e. keep k/(k+1) ratio of previous flow) so that the following shortes path will be receiving 1/(k+1) flow
					g_link_vector[l].flow_volume_per_period[tau] = g_link_vector[l].flow_volume_per_period[tau] * (float(iteration_index) / float(iteration_index + 1));
				}

			}
		}

	}
}

void g_reset_and_update_link_volume_based_on_columns(int number_of_links, int iteration_index, bool b_self_reducing_path_volume)
{

	int tau;
	int at;
	int l;
	for (l = 0; l < number_of_links; l++)
			{
				for (tau = 0; tau < assignment.g_number_of_demand_periods; tau++)
				{
					g_link_vector[l].flow_volume_per_period[tau] = 0; // used in travel time calculation
					g_link_vector[l].queue_length_perslot[tau] = 0;  // reserved for BPR-X
					g_link_vector[l].resource_per_period[tau] = g_link_vector[l].VDF_period[tau].ruc_base_resource; // base as the reference value

					for (at = 0; at < assignment.g_AgentTypeVector.size(); at++)
					{
						g_link_vector[l].volume_per_period_per_at[tau][at] = 0;
						g_link_vector[l].resource_per_period_per_at[tau][at] = 0;
					}

				}
			}

		if(iteration_index>=0)
		{

			for (int at = 0; at < assignment.g_AgentTypeVector.size(); at++)  //m
			{
//#pragma omp parallel for

				std::map<int, CColumnPath>::iterator it;
				int o, d, tau;
				int zone_size = g_zone_vector.size();
				int tau_size = assignment.g_DemandPeriodVector.size();

				float link_volume_contributed_by_path_volume;

				int link_seq_no;
				float PCE_ratio;
				float CRU;
				int nl;

				std::map<int, CColumnPath>::iterator it_begin;
				std::map<int, CColumnPath>::iterator it_end;

				int column_vector_size;
				CColumnVector* p_column;

				for (o = 0; o < zone_size; o++)  // o
				for (d = 0; d < zone_size; d++) //d
				for (tau = 0; tau < tau_size; tau++)  //tau
				{
					p_column = &(assignment.g_column_pool[o][d][at][tau]);
					if (p_column->od_volume > 0)
					{
						// continuous: type 0


						column_vector_size = p_column->path_node_sequence_map.size();

						it_begin = p_column->path_node_sequence_map.begin();
						it_end = p_column->path_node_sequence_map.end();

						for (it = it_begin ; it != it_end; it++)
						{
							
							link_volume_contributed_by_path_volume = it->second.path_volume;  // assign all OD flow to this first path

							// add path volume to link volume
							for (nl = 0; nl < it->second.m_link_size; nl++)  // arc a
							{
								link_seq_no = it->second.path_link_vector[nl];
								

								// MSA updating for the existing column pools
								// if iteration_index = 0; then update no flow discount is used (for the column pool case)
								PCE_ratio = g_link_vector[link_seq_no].PCE_at[at];
								//#pragma omp critical
								{

								g_link_vector[link_seq_no].flow_volume_per_period[tau] += link_volume_contributed_by_path_volume * PCE_ratio;
								g_link_vector[link_seq_no].volume_per_period_per_at[tau][at] += link_volume_contributed_by_path_volume;  // pure volume, not consider PCE

									if (assignment.assignment_mode == 3)
									{
										CRU = g_link_vector[link_seq_no].CRU_at[at];
										g_link_vector[link_seq_no].resource_per_period[tau] += link_volume_contributed_by_path_volume * CRU;
										g_link_vector[link_seq_no].resource_per_period_per_at[tau][at] += link_volume_contributed_by_path_volume * CRU;
									}
								}
							}

							if(b_self_reducing_path_volume == true)
							{							//after link volumn "tally", self-deducting the path volume by 1/(k+1) (i.e. keep k/(k+1) ratio of previous flow) so that the following shortes path will be receiving 1/(k+1) flow
							it->second.path_volume = it->second.path_volume * (float(iteration_index) / float(iteration_index + 1));
							}

						}

						// discrete: type 1
						for (int ai = 0; ai < p_column->discrete_agent_path_vector.size(); ai++)
						{
							
							CAgentPath agent_path = p_column->discrete_agent_path_vector[ai];

							for (int nl = 0; nl< agent_path.path_link_sequence.size(); nl++)  // arc a
							{
								int link_seq_no = agent_path.path_link_sequence[nl];

								float volume = agent_path.volume ;  // equals to all or nothing assignment

//#pragma omp critical
								{

									float PCE_ratio = g_link_vector[link_seq_no].PCE_at[at];
									g_link_vector[link_seq_no].flow_volume_per_period[tau] += volume * PCE_ratio;
									g_link_vector[link_seq_no].volume_per_period_per_at[tau][at] += volume;  // pure volume, not consider PCE


									float CRU = g_link_vector[link_seq_no].CRU_at[at];
									g_link_vector[link_seq_no].resource_per_period[tau] += volume * CRU;
									g_link_vector[link_seq_no].resource_per_period_per_at[tau][at] += volume * CRU;
								}

							}
						}
					}
				}
			

			}
		}

}

void update_link_travel_time_and_cost()
{
#pragma omp parallel for
	for (int l = 0; l < g_link_vector.size(); l++)
	{
		int tau;
		// step 1: travel time based on VDF
		g_link_vector[l].CalculateTD_VDFunction();
		

		for (tau = 0; tau < assignment.g_DemandPeriodVector.size(); tau++)
		{
			if (g_debugging_flag>=2 && assignment.g_pFileDebugLog != NULL)
				fprintf(assignment.g_pFileDebugLog, "Update link resource: link %d->%d: tau = %d, volume = %.2f, travel time = %.2f, resource = %.3f\n",

					g_node_vector[g_link_vector[l].from_node_seq_no].node_id,
					g_node_vector[g_link_vector[l].to_node_seq_no].node_id,
					tau,
					g_link_vector[l].flow_volume_per_period [tau],
					g_link_vector[l].travel_time_per_period[tau],
					g_link_vector[l].resource_per_period[tau]);
					

			for (int at = 0; at < assignment.g_AgentTypeVector.size(); at++)
			{

				float PCE_agent_type = assignment.g_AgentTypeVector[at].PCE_link_type_map[g_link_vector[l].link_type];

				// step 2: marginal cost for SO
				g_link_vector[l].calculate_marginal_cost_for_agent_type(tau, at, PCE_agent_type);

				float CRU_agent_type = assignment.g_AgentTypeVector[at].CRU_link_type_map[g_link_vector[l].link_type];

				// setp 3: penalty for resource constraints 
				g_link_vector[l].calculate_penalty_for_agent_type(tau, at, CRU_agent_type);


				if (g_debugging_flag>=2 && assignment.assignment_mode >=2 &&assignment.g_pFileDebugLog != NULL)
					fprintf(assignment.g_pFileDebugLog, "Update link cost: link %d->%d: tau = %d, at = %d, travel_marginal =  %.3f; penalty derivative = %.3f\n",

						g_node_vector[g_link_vector[l].from_node_seq_no].node_id,
						g_node_vector[g_link_vector[l].to_node_seq_no].node_id,
						tau, at,
						g_link_vector[l].travel_marginal_cost_per_period[tau][at],
						g_link_vector[l].exterior_penalty_derivative_per_period[tau][at]);

			}
		}
	}
}


void g_reset_and_update_Gauss_Seidel_link_volume_and_cost(int number_of_links, int at_current, int o_current, int d_current, int tau_current, int ai_current)
{
	for (int l = 0; l < number_of_links; l++)
	{
		for (int tau = 0; tau < assignment.g_number_of_demand_periods; tau++)
		{
			g_link_vector[l].resource_per_period[tau] = g_link_vector[l].VDF_period[tau].ruc_base_resource; // base as the reference value
		}
	}


		for (int at = 0; at < assignment.g_AgentTypeVector.size(); at++)  //m
		{
			for (int o = 0; o < g_zone_vector.size(); o++)  // o
				for (int d = 0; d < g_zone_vector.size(); d++) //d

					for (int tau = 0; tau < assignment.g_DemandPeriodVector.size(); tau++)  //tau
					{
						CColumnVector* p_column = &(assignment.g_column_pool[o][d][at][tau]);
						if (p_column->od_volume > 0)
						{
							
							// continuous: type 0
							std::map<int, CColumnPath>::iterator it;

							int column_vector_size = p_column->path_node_sequence_map.size();

							for (it = p_column->path_node_sequence_map.begin(); //k
								it != p_column->path_node_sequence_map.end(); it++)
							{
								float link_volume_contributed_by_path_volume = 0;

								link_volume_contributed_by_path_volume = it->second.path_volume;  // assign all OD flow to this first path

								// add path volume to link volume
								for (int nl = 0; nl < it->second.m_link_size; nl++)  // arc a
								{
									int link_seq_no = it->second.path_link_vector[nl];

									float CRU = assignment.g_AgentTypeVector[at].CRU_link_type_map[g_link_vector[link_seq_no].link_type];
									g_link_vector[link_seq_no].resource_per_period[tau] += link_volume_contributed_by_path_volume * CRU;
									g_link_vector[link_seq_no].resource_per_period_per_at[tau][at] += link_volume_contributed_by_path_volume * CRU;

								}

							}

							// discrete: type 1
							for (int ai = 0; ai < p_column->discrete_agent_path_vector.size(); ai++)
							{

								if (at_current == at && o_current == o && d_current == d && tau_current == tau && ai_current == ai)
								{
									continue; // skip
								}

								CAgentPath agent_path = p_column->discrete_agent_path_vector[ai];

								for (int nl = 0; nl < agent_path.path_link_sequence.size(); nl++)  // arc a
								{
									int link_seq_no = agent_path.path_link_sequence[nl];

									float volume = agent_path.volume;  // equals to all or nothing assignment

									float CRU = assignment.g_AgentTypeVector[at].CRU_link_type_map[g_link_vector[link_seq_no].link_type];
									g_link_vector[link_seq_no].resource_per_period[tau] += volume * CRU;
									g_link_vector[link_seq_no].resource_per_period_per_at[tau][at] += volume * CRU;


								}
							}
						}
					}


		
	}



	for (int l = 0; l < g_link_vector.size(); l++)
	{
		// step 1: travel time based on VDF
		g_link_vector[l].CalculateTD_VDFunction();
		float CRU_agent_type = assignment.g_AgentTypeVector[at_current].CRU_link_type_map[g_link_vector[l].link_type];
	// setp 3: penalty for resource constraints 
		g_link_vector[l].calculate_Gauss_Seidel_penalty_for_agent_type(tau_current, at_current, CRU_agent_type);

	}
}
void g_update_gradient_cost_and_assigned_flow_in_column_pool(Assignment& assignment, int inner_iteration_number)
{
	float total_gap = 0;
	float total_relative_gap = 0;
	float total_gap_count = 0;
	g_reset_and_update_link_volume_based_on_columns(g_link_vector.size(), inner_iteration_number, false);  // we can have a recursive formulat to reupdate the current link volume by a factor of k/(k+1), and use the newly generated path flow to add the additional 1/(k+1)
	//step 4: based on newly calculated path volumn, update volume based travel time, and update volume based resource balance, update gradie
	update_link_travel_time_and_cost();
	// step 0


	//step 1: calculate shortest path at inner iteration of column flow updating
#pragma omp parallel for
	for (int o = 0; o < g_zone_vector.size(); o++)  // o
	{
		int d, at, tau;
		CColumnVector* p_column;
		std::map<int, CColumnPath>::iterator it, it_begin, it_end;
		int column_vector_size;

		float least_gradient_cost = 999999;
		int least_gradient_cost_path_seq_no = -1;
		int least_gradient_cost_path_node_sum_index = -1;
		int path_seq_count = 0;


		float path_cost = 0;
		float path_gradient_cost = 0;
		float path_distance = 0;
		float path_travel_time = 0;
		int nl;
		int link_seq_no;

		float link_travel_time;
		float total_switched_out_path_volume = 0;

		float step_size = 0;
		float previous_path_volume = 0;

		for (d = 0; d < g_zone_vector.size(); d++) //d
			for (at = 0; at < assignment.g_AgentTypeVector.size(); at++)  //m
				for (tau = 0; tau < assignment.g_DemandPeriodVector.size(); tau++)  //tau
				{
					p_column = &(assignment.g_column_pool[o][d][at][tau]);
					if (p_column->od_volume > 0)
					{

						column_vector_size = p_column->path_node_sequence_map.size();


						// scan through the map with different node sum for different paths
						/// step 1: update gradient cost for each column path
						//if (o = 7 && d == 15)
						//{

						//	if (assignment.g_pFileDebugLog != NULL)
						//		fprintf(assignment.g_pFileDebugLog, "CU: iteration %d: total_gap=, %f,total_relative_gap,%f,\n", inner_iteration_number, total_gap, total_gap / max(0.00001, total_gap_count));
						//}
						 least_gradient_cost = 999999;
						 least_gradient_cost_path_seq_no = -1;
						 least_gradient_cost_path_node_sum_index = -1;
						 path_seq_count = 0;

						 it_begin = p_column->path_node_sequence_map.begin();
						 it_end = p_column->path_node_sequence_map.end();

						for (it = it_begin; it != it_end; it++)
						{


							 path_cost = 0;
							 path_gradient_cost = 0;
							 path_distance = 0;
							 path_travel_time = 0;

							for (nl = 0; nl < it->second.m_link_size; nl++)  // arc a
							{
								link_seq_no = it->second.path_link_vector[nl];
								path_cost += g_link_vector[link_seq_no].cost;
								path_distance += g_link_vector[link_seq_no].length;
								link_travel_time = g_link_vector[link_seq_no].travel_time_per_period[tau];
								path_travel_time += link_travel_time;

								path_gradient_cost += g_link_vector[link_seq_no].get_generalized_first_order_gradient_cost_of_second_order_loss_for_agent_type(tau, at);
							}


							it->second.path_cost = path_cost;
							it->second.path_travel_time = path_travel_time;
							it->second.path_gradient_cost = path_gradient_cost;

							if (column_vector_size == 1)  // only one path
							{
								total_gap_count += (it->second.path_gradient_cost * it->second.path_volume);
								break;
							}


							if (path_gradient_cost < least_gradient_cost)
							{
								least_gradient_cost = path_gradient_cost;
								least_gradient_cost_path_seq_no = it->second.path_seq_no;
								least_gradient_cost_path_node_sum_index = it->first;
							}


						}


						if (column_vector_size >= 2)
						{


							// step 2: calculate gradient cost difference for each column path
							total_switched_out_path_volume = 0;
							for (it = it_begin; it != it_end; it++)
							{
								if (it->second.path_seq_no != least_gradient_cost_path_seq_no)  //for non-least cost path
								{

									it->second.path_gradient_cost_difference = it->second.path_gradient_cost - least_gradient_cost;
									it->second.path_gradient_cost_relative_difference = it->second.path_gradient_cost_difference / max(0.0001, least_gradient_cost);

									total_gap += (it->second.path_gradient_cost_difference * it->second.path_volume);
									total_gap_count += (it->second.path_gradient_cost * it->second.path_volume);

									step_size = 1.0 / (inner_iteration_number + 2) * p_column->od_volume;

									previous_path_volume = it->second.path_volume;


									//recall that it->second.path_gradient_cost_difference >=0 
									// step 3.1: shift flow from nonshortest path to shortest path
									it->second.path_volume = max(0, it->second.path_volume - step_size * it->second.path_gradient_cost_relative_difference);


									//we use min(step_size to ensure a path is not switching more than 1/n proportion of flow 
									it->second.path_switch_volume = (previous_path_volume - it->second.path_volume);
									total_switched_out_path_volume += (previous_path_volume - it->second.path_volume);

								}

							}

							//step 3.2 consider least cost path, receive all volume shifted from non-shortest path
							if (least_gradient_cost_path_seq_no != -1)
							{
								if (assignment.g_AgentTypeVector[at].flow_type == 0)  // continuous
								{
									p_column->path_node_sequence_map[least_gradient_cost_path_node_sum_index].path_volume += total_switched_out_path_volume;
									total_gap_count += (p_column->path_node_sequence_map[least_gradient_cost_path_node_sum_index].path_gradient_cost *
										p_column->path_node_sequence_map[least_gradient_cost_path_node_sum_index].path_volume);
								}

							}

						}


					}

				}
	}
	if(assignment.g_pFileDebugLog != NULL)
	fprintf(assignment.g_pFileDebugLog, "CU: iteration %d: total_gap=, %f,total_relative_gap=, %f,\n", inner_iteration_number, total_gap, total_gap / max(0.0001, total_gap_count));

}


void g_column_pool_optimization(Assignment& assignment, int column_updating_iterations)
{
	// column_updating_iterations is internal numbers of column updating
	for (int n = 0; n < column_updating_iterations; n++)
	{ 
		cout << "column pool updating: no.  " << n << endl;
		g_update_gradient_cost_and_assigned_flow_in_column_pool(assignment, n);
	}
}

char str_buffer[STRING_LENGTH_PER_LINE];

void g_output_simulation_result(Assignment& assignment)
{
	cout << "writing link_performance.csv.." << endl;

	int b_debug_detail_flag = 0;
	FILE* g_pFileLinkMOE = NULL;
	fopen_ss(&g_pFileLinkMOE,"link_performance.csv", "w");
	if (g_pFileLinkMOE == NULL)
	{
		cout << "File link_performance.csv cannot be opened." << endl;
		g_ProgramStop();
	}
	else
	{
		// Option 2: BPR-X function
		fprintf(g_pFileLinkMOE, "road_link_id,from_node_id,to_node_id,time_period,volume,travel_time,speed,VOC,");

		for (int at = 0; at < assignment.g_AgentTypeVector.size(); at++)
		{
			fprintf(g_pFileLinkMOE, "volume_at_%s,", assignment.g_AgentTypeVector[at].agent_type.c_str());
		}

		fprintf(g_pFileLinkMOE, "resource_balance,");

		fprintf(g_pFileLinkMOE, "notes\n");
		

		for (int l = 0; l < g_link_vector.size(); l++) //Initialization for all nodes
		{
			for (int tau = 0; tau < assignment.g_number_of_demand_periods; tau++)
			{
				float speed = g_link_vector[l].length / (max(0.001,g_link_vector[l].VDF_period[tau].avg_travel_time )/ 60.0);
				fprintf(g_pFileLinkMOE, "%d,%d,%d,%s,%.3f,%.3f,%.3f,%.3f,",
					g_link_vector[l].link_id,

					g_node_vector[g_link_vector[l].from_node_seq_no].node_id,
					g_node_vector[g_link_vector[l].to_node_seq_no].node_id,

					assignment.g_DemandPeriodVector[tau].time_period.c_str(),
					g_link_vector[l].flow_volume_per_period[tau],
					g_link_vector[l].VDF_period[tau].avg_travel_time,
					speed,  /* 60.0 is used to convert min to hour */
					g_link_vector[l].VDF_period[tau].VOC);

				if (assignment.assignment_mode == 3)  // exterior panalty mode
				{ 
					for (int at = 0; at < assignment.g_AgentTypeVector.size(); at++)
					{ 
						fprintf(g_pFileLinkMOE, "%.3f,", g_link_vector[l].volume_per_period_per_at[tau][at]);
					}

					fprintf(g_pFileLinkMOE, "%.3f,", g_link_vector[l].resource_per_period[tau]);
				}
				else
				{  /// no resource output
					for (int at = 0; at < assignment.g_AgentTypeVector.size(); at++)
					{
						fprintf(g_pFileLinkMOE, ",");
					}

					fprintf(g_pFileLinkMOE, ",");

				}

			}



				fprintf(g_pFileLinkMOE, "period-based\n");
		}

		//for (int l = 0; l < g_link_vector.size(); l++) //Initialization for all nodes
		//{
		//	for (int tau = 0; tau < assignment.g_number_of_demand_periods; tau++)
		//	{
		//		
		//			int starting_time = g_link_vector[l].VDF_period[tau].starting_time_slot_no;
		//			int ending_time = g_link_vector[l].VDF_period[tau].ending_time_slot_no;

		//			for (int t = 0; t <= ending_time - starting_time; t++)
		//			{
		//				fprintf(g_pFileLinkMOE, "%s,%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%s\n",

		//					g_link_vector[l].link_id.c_str(),
		//					g_node_vector[g_link_vector[l].from_node_seq_no].node_id.c_str(),
		//					g_node_vector[g_link_vector[l].to_node_seq_no].node_id.c_str(),
		//					t,
		//					g_link_vector[l].VDF_period[tau].discharge_rate[t],
		//					g_link_vector[l].VDF_period[tau].travel_time[t],
		//					g_link_vector[l].length / g_link_vector[l].VDF_period[tau].travel_time[t] * 60.0,
		//					g_link_vector[l].VDF_period[tau].congestion_period_P,
		//					"timeslot-dependent");
		//			}

		//		}

		//}

		


	fclose(g_pFileLinkMOE);
	}

	cout << "writing agent.csv.." << endl;

	float path_time_vector[_MAX_LINK_SIZE_IN_A_PATH];
	FILE* g_pFileODMOE = NULL;
	fopen_ss(&g_pFileODMOE,"agent.csv", "w");
	if (g_pFileODMOE == NULL)
	{
		cout << "File agent.csv cannot be opened." << endl;
		g_ProgramStop();
	}
	else
	{
		fprintf(g_pFileODMOE, "agent_id,o_zone_id,d_zone_id,path_id,o_node_id,d_node_id,agent_type,demand_period,volume,cost,travel_time,distance,opti_cost,oc_diff,relative_diff,node_sequence,link_sequence,time_sequence,time_decimal_sequence,\n");


		int count = 1;

		clock_t start_t, end_t, total_t;
		start_t = clock();
		clock_t iteration_t;



		int buffer_len;

		int agent_type_size = assignment.g_AgentTypeVector.size();
		int zone_size = g_zone_vector.size();
		int o, at, d, tau;

		int demand_period_size = assignment.g_DemandPeriodVector.size();

		CColumnVector* p_column;

		float path_cost = 0;
		float path_distance = 0;
		float path_travel_time = 0;
		float time_stamp = 0;
		
		std::map<int, CColumnPath>::iterator it, it_begin, it_end;

		for (o = 0; o < zone_size; o++)
		{ 
			if (o % 100 == 0)
			{
				cout << "writing " << o << "  zones " << endl;
			}

			for (at = 0; at < agent_type_size; at++)
			for (d = 0; d < zone_size; d++)
					for (tau = 0; tau < demand_period_size; tau++)
					{
						p_column = &(assignment.g_column_pool[o][d][at][tau]);
						if (p_column->od_volume > 0)
						{


							time_stamp = (assignment.g_DemandPeriodVector[tau].starting_time_slot_no + assignment.g_DemandPeriodVector[tau].ending_time_slot_no) / 2.0 * MIN_PER_TIMESLOT;

							// scan through the map with different node sum for different continuous paths
							it_begin = p_column->path_node_sequence_map.begin();
							it_end = p_column->path_node_sequence_map.end();

							for (it = it_begin;it != it_end; it++)
							{
								if (count%100000 ==0)
								{ 
									end_t = clock();
									iteration_t = end_t - start_t;
								cout << "writing " << count/1000 << "K agents with CPU time " << iteration_t / 1000.0 << " s" << endl;
								}

								path_cost = 0;
								path_distance = 0;
								path_travel_time = 0;

								path_time_vector[0] = time_stamp;


								for (int nl = 0; nl < it->second.m_link_size; nl++)  // arc a
								{
									int link_seq_no = it->second.path_link_vector[nl];
									path_cost += g_link_vector[link_seq_no].cost;
									path_distance += g_link_vector[link_seq_no].length;
									float link_travel_time = g_link_vector[link_seq_no].travel_time_per_period[tau];
									path_travel_time += link_travel_time;
									time_stamp += link_travel_time;
									path_time_vector[nl+1] = time_stamp;
								}

								buffer_len = 0;
								buffer_len = sprintf(str_buffer, "%d,%d,%d,%d,%d,%d,%s,%s,%.2f,%.1f,%.4f,%.4f,%.4f,%.4f,%.4f,",
									count,
									g_zone_vector[o].zone_id,
									g_zone_vector[d].zone_id,
									it->second.path_seq_no,
									g_node_vector[g_zone_vector[o].node_seq_no].node_id,
									g_node_vector[g_zone_vector[d].node_seq_no].node_id,
									assignment.g_AgentTypeVector[at].agent_type.c_str(),
									assignment.g_DemandPeriodVector[tau].demand_period.c_str(),
									it->second.path_volume,
									path_cost,
									path_travel_time,
									path_distance,
									it->second.path_gradient_cost,
									it->second.path_gradient_cost_difference,
									it->second.path_gradient_cost_relative_difference * it->second.path_volume
									);

								/* Format and print various data */

								for (int ni = 0; ni <it->second.m_node_size; ni ++)
								{
									buffer_len += sprintf(str_buffer + buffer_len, "%d;", g_node_vector[it->second.path_node_vector[ni]].node_id);
								}

								buffer_len += sprintf(str_buffer+ buffer_len, ",");

								for (int nl = 0; nl < it->second.m_link_size; nl++)
								{
									int link_seq_no = it->second.path_link_vector[nl];
									buffer_len += sprintf(str_buffer + buffer_len, "%d;", g_link_vector[link_seq_no].link_id);

								}
								buffer_len += sprintf(str_buffer + buffer_len, ",");

								for (int nt = 0; nt < it->second.m_link_size+1; nt++)
								{
									buffer_len += sprintf(str_buffer + buffer_len, "%s;", g_time_coding(path_time_vector[nt]).c_str());

								}
								buffer_len += sprintf(str_buffer + buffer_len, ",");

								for (int nt = 0; nt < it->second.m_link_size+1; nt++)
								{
									buffer_len += sprintf(str_buffer + buffer_len, "%.2f;", path_time_vector[nt]);
								}

								buffer_len += sprintf(str_buffer + buffer_len, "\n");

								if (buffer_len >= STRING_LENGTH_PER_LINE - 1)
								{
									cout << "Error: buffer_len >= STRING_LENGTH_PER_LINE." << endl;
									g_ProgramStop();
								}


								fprintf(g_pFileODMOE, "%s", str_buffer, buffer_len);
							count++;

							}

								//********************************************************************************************************************************
																// scan through the map with different node sum for different discrete_agent paths
								for (int ai = 0; ai < p_column->discrete_agent_path_vector.size(); ai++)
								{
									CAgentPath agent_path = p_column->discrete_agent_path_vector[ai];
									{

										float path_cost = 0;
										float path_distance = 0;
										float path_travel_time = 0;

										vector<float> path_time_vector;

										float time_stamp = (assignment.g_DemandPeriodVector[tau].starting_time_slot_no + assignment.g_DemandPeriodVector[tau].ending_time_slot_no) / 2.0 * MIN_PER_TIMESLOT;

										path_time_vector.push_back(time_stamp);

										for (int nl = 0; nl < agent_path.path_link_sequence.size(); nl++)
										{
											int link_seq_no = agent_path.path_link_sequence[nl];
											path_cost += g_link_vector[link_seq_no].cost;
											path_distance += g_link_vector[link_seq_no].length;
											float link_travel_time = g_link_vector[link_seq_no].travel_time_per_period[tau];
											path_travel_time += link_travel_time;
											time_stamp += link_travel_time;
											path_time_vector.push_back(time_stamp);
										}


										fprintf(g_pFileODMOE, "%d,%d,%d,%d,%d,%d,%s,%s,%.1f,%.1f,%.1f,%.4f,0,0,0,",
											count,
											g_zone_vector[o].zone_id,
											g_zone_vector[d].zone_id,
											agent_path.path_id,
											g_node_vector[g_zone_vector[o].node_seq_no].node_id,
											g_node_vector[g_zone_vector[d].node_seq_no].node_id,
											assignment.g_AgentTypeVector[at].agent_type.c_str(),
											assignment.g_DemandPeriodVector[tau].demand_period.c_str(),
											agent_path.volume,
											path_cost,
											path_travel_time,
											path_distance);


										int to_node_id;

										for (int nl = 0; nl < agent_path.path_link_sequence.size(); nl++)
										{
											int link_seq_no = agent_path.path_link_sequence[nl];
											to_node_id = g_node_vector[g_link_vector[link_seq_no].to_node_seq_no].node_id;
											fprintf(g_pFileODMOE, "%d;", g_node_vector[g_link_vector[link_seq_no].from_node_seq_no].node_id);
										}
										if (agent_path.path_link_sequence.size() >= 1)
											fprintf(g_pFileODMOE, "%d;", to_node_id);

										fprintf(g_pFileODMOE, ",");
										for (int nl = 0; nl < agent_path.path_link_sequence.size(); nl++)
										{
											int link_seq_no = agent_path.path_link_sequence[nl];
											fprintf(g_pFileODMOE, "%d;", g_link_vector[link_seq_no].link_id);

										}
										fprintf(g_pFileODMOE, ",");

										for (int nt = 0; nt < path_time_vector.size(); nt++)
										{
											fprintf(g_pFileODMOE, "%s;", g_time_coding(path_time_vector[nt]).c_str());
										}
										fprintf(g_pFileODMOE, ",");

										for (int nt = 0; nt < path_time_vector.size(); nt++)
										{
											fprintf(g_pFileODMOE, "%.2f;", path_time_vector[nt]);
										}

										fprintf(g_pFileODMOE, "\n");

										count++;
									}

								}
							

						}
					}
		}
	}
	fclose(g_pFileODMOE);
}

//***
// major function 1:  allocate memory and initialize the data
// void AllocateMemory(int number_of_nodes)
//
//major function 2: // time-dependent label correcting algorithm with double queue implementation
//int optimal_label_correcting(int origin_node, int destination_node, int departure_time, int g_debugging_flag, FILE* g_pFileDebugLog, Assignment& assignment, int time_period_no = 0, int agent_type = 1, float VOT = 10)

//	//major function: update the cost for each node at each SP tree, using a stack from the origin structure 
//int tree_cost_updating(int origin_node, int departure_time, int g_debugging_flag, FILE* g_pFileDebugLog, Assignment& assignment, int time_period_no = 0, int agent_type = 1)

//***

// The one and only application object


int g_number_of_CPU_threads()
{
	int number_of_threads = omp_get_max_threads();

	int max_number_of_threads = 4000;

	if (number_of_threads > max_number_of_threads)
		number_of_threads = max_number_of_threads;

	return number_of_threads;

}


void g_assign_computing_tasks_to_memory_blocks(Assignment& assignment)
{
	//fprintf(g_pFileDebugLog, "-------g_assign_computing_tasks_to_memory_blocks-------\n");
	// step 2: assign node to thread
	for (int at = 0; at < assignment.g_AgentTypeVector.size(); at++)
	{
		if (assignment.g_AgentTypeVector[at].flow_type == 0)  // fluid based
		{ 
		for (int z = 0; z < g_zone_vector.size(); z++)  //assign all nodes to the corresponding thread
		{
					for (int tau = 0; tau < assignment.g_DemandPeriodVector.size(); tau++)
					{
						if (assignment.g_origin_demand_array[z][at][tau] > 0) // with feasible flow
						{
							//fprintf(g_pFileDebugLog, "%f\n",g_origin_demand_array[zone_seq_no][at][tau]);
								//cout << assignment.g_origin_demand_array[zone_seq_no][at][tau] << endl;

							if (z < assignment.g_number_of_memory_blocks)
							{
								NetworkForSP* p_NetworkForSP = new NetworkForSP();

								p_NetworkForSP->m_origin_node_vector.push_back(g_zone_vector[z].node_seq_no);
								p_NetworkForSP->m_origin_zone_seq_no_vector.push_back(z);

								p_NetworkForSP->m_agent_type_no = at;
								p_NetworkForSP->tau = tau;
								p_NetworkForSP->AllocateMemory(assignment.g_number_of_nodes, assignment.g_number_of_links);

								if (z % 100 == 0)
									cout << "Allocating network memory block for zone " << z << endl;

								g_NetworkForSP_vector.push_back(p_NetworkForSP);
							}
							else  // zone seq no is greater than g_number_of_memory_blocks
							{
								//get the corresponding memory block seq no
								int memory_block_no = z % assignment.g_number_of_memory_blocks;  // take residual of memory block size to map a zone no to a memory block no.
								NetworkForSP* p_NetworkForSP = g_NetworkForSP_vector[memory_block_no];
								p_NetworkForSP->m_origin_node_vector.push_back(g_zone_vector[z].node_seq_no);
								p_NetworkForSP->m_origin_zone_seq_no_vector.push_back(z);


							}


					}
				}


			}
		}
	
	}

}

void g_reset_link_volume_for_all_processors()
{
#pragma omp parallel for
	for (int ProcessID = 0; ProcessID < g_NetworkForSP_vector.size(); ProcessID++)
	{
		for (int i = 0; i < assignment.g_number_of_links; i++) //Initialization for all non-origin nodes
		{
			g_NetworkForSP_vector[ProcessID] ->m_link_flow_volume_array[i] = 0;
		}
	}

}


void g_fetch_link_volume_for_all_processors()
{
#pragma omp parallel for
	for (int l = 0; l < g_link_vector.size(); l++)
	{
		// step 1: travel time based on VDF
	
			for (int ProcessID = 0; ProcessID < g_NetworkForSP_vector.size(); ProcessID++) 
			{
				int tau = g_NetworkForSP_vector[ProcessID]->tau;
				g_link_vector[l].flow_volume_per_period[tau] += g_NetworkForSP_vector[ProcessID]->m_link_flow_volume_array[l];
			}
	}
}


//major function: update the cost for each node at each SP tree, using a stack from the origin structure 

void NetworkForSP::backtrace_shortest_path_tree(Assignment& assignment, int iteration_number_outterloop, int o_node_index)
{

		int origin_node = m_origin_node_vector[o_node_index]; // assigned no
		int m_origin_zone_seq_no = m_origin_zone_seq_no_vector[o_node_index]; // assigned no


		int departure_time = tau;

		int agent_type = m_agent_type_no;


		if (g_node_vector[origin_node].m_outgoing_link_seq_no_vector.size() == 0)
		{
			return;
		}

		// given,  m_node_label_cost[i]; is the gradient cost , for this tree's, from its origin to the destination node i'. 

		//	fprintf(g_pFileDebugLog, "------------START: origin:  %d  ; Departure time: %d  ; demand type:  %d  --------------\n", origin_node + 1, departure_time, agent_type);
		float k_path_prob = 1;
		if (assignment.g_AgentTypeVector[agent_type].flow_type == 0)  // continuous
		{
			k_path_prob = float(1) / float(iteration_number_outterloop + 1);  //XZ: use default value as MSA, this is equivalent to 1/(n+1)
			// MSA to distribute the continuous flow
		}
		// to do, this is for each nth tree. 

		//change of path flow is a function of cost gap (the updated node cost for the path generated at the previous iteration -m_node_label_cost[i] at the current iteration)
	   // current path flow - change of path flow, 
	   // new propability for flow staying at this path
	   // for current shortest path, collect all the switched path from the other shortest paths for this ODK pair.
	   // check demand flow balance constraints 

		int num = 0;
		int number_of_nodes = assignment.g_number_of_nodes;
		int number_of_links = assignment.g_number_of_links;
		int l_node_size = 0;  // initialize local node size index
		int l_link_size = 0;
		int node_sum = 0;

		float path_travel_time = 0;
		float path_distance = 0;


		int current_node_seq_no = -1;  // destination node
		int current_link_seq_no = -1;
		int destination_zone_seq_no;
		float ODvolume, volume;
		CColumnVector* pColumnVector;

		for (int i = 0; i < number_of_nodes; i++)
		{

			if (g_node_vector[i].zone_id >= 1)
			{

				if (i == origin_node) // no within zone demand
					continue;
				//			fprintf(g_pFileDebugLog, "--------origin  %d ; destination node: %d ; (zone: %d) -------\n", origin_node + 1, i+1, g_node_vector[i].zone_id);
				//fprintf(g_pFileDebugLog, "--------iteration number outterloop  %d ;  -------\n", iteration_number_outterloop);
				destination_zone_seq_no = assignment.g_zoneid_to_zone_seq_no_mapping[g_node_vector[i].zone_id];


				 pColumnVector = &(assignment.g_column_pool[m_origin_zone_seq_no][destination_zone_seq_no][agent_type][tau]);
				 ODvolume = pColumnVector->od_volume;


				 volume = ODvolume * k_path_prob;
				// this is contributed path volume from OD flow (O, D, k, per time period


				if (ODvolume > 0.000001)
				{
					l_node_size = 0;  // initialize local node size index
					l_link_size = 0;
					node_sum = 0;

					path_travel_time = 0;
					path_distance = 0;


					current_node_seq_no = i;  // destination node
					current_link_seq_no = -1;

					// backtrace the sp tree from the destination to the root (at origin) 
					while (current_node_seq_no >= 0 && current_node_seq_no < number_of_nodes)
					{

						temp_path_node_vector[l_node_size++] = current_node_seq_no;

						if (l_node_size >= temp_path_node_vector_size)
						{
							cout << "Error: l_node_size >= temp_path_node_vector_size" << endl;
							g_ProgramStop();
						}


						if (current_node_seq_no >= 0 && current_node_seq_no < number_of_nodes)  // this is valid node 
						{
							node_sum += current_node_seq_no;
							current_link_seq_no = m_link_predecessor[current_node_seq_no];

							// fetch m_link_predecessor to mark the link volume

							if (current_link_seq_no >= 0 && current_link_seq_no < number_of_links)
							{
								temp_path_link_vector[l_link_size++] = current_link_seq_no;

								if (assignment.assignment_mode == 0) // pure link based computing mode
								{

									m_link_flow_volume_array[current_link_seq_no]+= volume;  // this is critical for parallel computing as we can write the volume to data
								}

								//path_travel_time += g_link_vector[current_link_seq_no].travel_time_per_period[tau];
								//path_distance += g_link_vector[current_link_seq_no].length;

							}
						}
						current_node_seq_no = m_node_predecessor[current_node_seq_no];  // update node seq no	
					}
					//fprintf(g_pFileDebugLog, "\n");

					// we obtain the cost, time, distance from the last tree-k 
					if(assignment.assignment_mode >=1) // column based mode
					{
						if (pColumnVector->path_node_sequence_map.find(node_sum) == assignment.g_column_pool[m_origin_zone_seq_no][destination_zone_seq_no][agent_type][tau].path_node_sequence_map.end())
							// we cannot find a path with the same node sum, so we need to add this path into the map, 
						{
							// add this unique path
							int path_count = pColumnVector->path_node_sequence_map.size();
							pColumnVector->path_node_sequence_map[node_sum].path_seq_no = path_count;
							pColumnVector->path_node_sequence_map[node_sum].path_volume = 0;
							//assignment.g_column_pool[m_origin_zone_seq_no][destination_zone_seq_no][agent_type][tau].time = m_label_time_array[i];
							//assignment.g_column_pool[m_origin_zone_seq_no][destination_zone_seq_no][agent_type][tau].path_node_sequence_map[node_sum].path_distance = m_label_distance_array[i];
							pColumnVector->path_node_sequence_map[node_sum].path_cost = m_node_label_cost[i];

							pColumnVector->path_node_sequence_map[node_sum].AllocateVector(
								l_node_size,
								temp_path_node_vector,
								l_link_size,
								temp_path_link_vector);


						}

						pColumnVector->path_node_sequence_map[node_sum].path_volume += volume;

					}


				}
			}

		}

	
}

void  CLink::CalculateTD_VDFunction()
{
	for (int tau = 0; tau < assignment.g_number_of_demand_periods; tau++)
	{
		float starting_time_slot_no = assignment.g_DemandPeriodVector[tau].starting_time_slot_no;
		float ending_time_slot_no = assignment.g_DemandPeriodVector[tau].ending_time_slot_no;
		travel_time_per_period[tau] = VDF_period[tau].PerformBPR(flow_volume_per_period[tau]);
		//travel_time_per_period[tau] = VDF_period[tau].PerformBPR_X(flow_volume_per_period[tau]);

	}
}


double network_assignment(int iteration_number, int assignment_mode, int column_updating_iterations)
{

	//fopen_ss(&g_pFileOutputLog, "output_solution.csv", "w");
	//if (g_pFileOutputLog == NULL)
	//{
	//	cout << "File output_solution.csv cannot be opened." << endl;
	//	g_ProgramStop();
	//}

	if (assignment.g_pFileDebugLog == NULL)
		fopen_ss(&assignment.g_pFileDebugLog, "STALite_log.txt", "w");

	//{
	//	cout << "File output_solution.csv cannot be opened." << endl;
	//	g_ProgramStop();
	//}


	assignment.g_number_of_K_paths = iteration_number; // k iterations for column generation
	assignment.assignment_mode = assignment_mode; // 0: link UE: 1: path UE, 2: Path SO, 3: path resource constraints 
	if (assignment.assignment_mode == 0)
		column_updating_iterations = 0;

	// step 1: read input data of network / demand tables / Toll
	g_ReadInputData(assignment);

	g_ReadDemandFileBasedOnDemandFileList(assignment);


	//step 2: allocate memory and assign computing tasks
	g_assign_computing_tasks_to_memory_blocks(assignment); // static cost based label correcting 

	// definte timestamps
	clock_t start_t, end_t, total_t;
	start_t = clock();
	clock_t iteration_t, cumulative_lc, cumulative_cp, cumulative_lu;

	//step 3: column generation stage: find shortest path and update path cost of tree using TD travel time
	for (int iteration_number = 0; iteration_number < assignment.g_number_of_K_paths; iteration_number++)
	{
		end_t = clock();
		iteration_t = end_t - start_t;
		cout << "assignment iteration = " << iteration_number << " with CPU time " << iteration_t / 1000.0 << " s" << endl;

		//TRACE("Loop 1: assignment iteration %d", iteration_number);
	   // step 3.1 update travel time and resource consumption		
		clock_t start_t_lu = clock();
		if (assignment.assignment_mode != 0)
		{
			g_reset_and_update_link_volume_based_on_columns(g_link_vector.size(), iteration_number, true);  // we can have a recursive formulat to reupdate the current link volume by a factor of k/(k+1), and use the newly generated path flow to add the additional 1/(k+1)
		}
		else
		{
			g_reset_link_volume_without_columns(g_link_vector.size(), iteration_number, true);
		}

		update_link_travel_time_and_cost();  // initialization at the first iteration of shortest path

		end_t = clock();
		iteration_t = end_t - start_t_lu;
		cout << "Link update with CPU time " << iteration_t / 1000.0 << " s; " << (end_t - start_t) / 1000.0 << " s" << endl;

		//****************************************//
		//step 3.2 computng block for continuous variables;

		if (assignment.assignment_mode == 0)  // link based computing mode, we have to collect link volume from all processors.
		{
			g_reset_link_volume_for_all_processors();
		}


		clock_t start_t_lc = clock();
		clock_t  start_t_cp = clock();

		cumulative_lc = 0;
		cumulative_cp = 0;
		cumulative_lu = 0;

#pragma omp parallel for  // step 3: C++ open mp automatically create n threads., each thread has its own computing thread on a cpu core 
		for (int ProcessID = 0; ProcessID < g_NetworkForSP_vector.size(); ProcessID++)
		{
			int agent_type_no = g_NetworkForSP_vector[ProcessID]->m_agent_type_no;
			if (assignment.g_AgentTypeVector[agent_type_no].flow_type == 0)
			{
				for (int o_node_index = 0; o_node_index < g_NetworkForSP_vector[ProcessID]->m_origin_node_vector.size(); o_node_index++)
				{
					start_t_lc = clock();
					g_NetworkForSP_vector[ProcessID]->optimal_label_correcting(assignment, iteration_number, o_node_index);
					end_t = clock();
					cumulative_lc += end_t - start_t_lc;


					start_t_cp = clock();
					g_NetworkForSP_vector[ProcessID]->backtrace_shortest_path_tree(assignment, iteration_number, o_node_index);
					end_t = clock();
					cumulative_cp += end_t - start_t_cp;

				}

			}

			// perform one to all shortest path tree calculation
		}

		if (assignment.assignment_mode == 0)  // link based computing mode, we have to collect link volume from all processors.
		{
			g_fetch_link_volume_for_all_processors();
		}

		cout << "LC with CPU time " << cumulative_lc / 1000.0 << " s; " << endl;
		cout << "column generation with CPU time " << cumulative_cp / 1000.0 << " s; " << endl;

		//****************************************//
		//step 3.3 computing block for discrete variables;

		if (iteration_number == 0)
		{
			g_RoutingNetwork.AllocateMemory(assignment.g_number_of_nodes, assignment.g_number_of_links);
		}
		// prepare network routing engine for dynamic programming based integer flow space time routing

		for (int at = 0; at < assignment.g_AgentTypeVector.size(); at++)  //m
			if (assignment.g_AgentTypeVector[at].flow_type == 2)  // we only take into account the MAS generated agent volume into the link volume and link resource in this second optimization stage.
			{
				for (int o = 0; o < g_zone_vector.size(); o++)  // o
					for (int d = 0; d < g_zone_vector.size(); d++) //d
						for (int tau = 0; tau < assignment.g_DemandPeriodVector.size(); tau++)  //tau
						{
							CColumnVector* p_column = &(assignment.g_column_pool[o][d][at][tau]);
							if (p_column->od_volume > 0)
							{
								for (int ai = 0; ai < p_column->discrete_agent_path_vector.size(); ai++)
								{
									CAgentPath agent_path = p_column->discrete_agent_path_vector[ai];
									{



										// internal step 1: test shortest path travel time 
										g_RoutingNetwork.m_origin_node_vector.push_back(agent_path.o_node_no);
										g_RoutingNetwork.m_origin_zone_seq_no_vector.push_back(agent_path.o_node_no);

										g_RoutingNetwork.m_agent_type_no = at;
										g_RoutingNetwork.tau = tau;
										float route_trip_time = g_RoutingNetwork.optimal_label_correcting(assignment, 0, 0, agent_path.d_node_no, true);

										//	g_reset_and_update_Gauss_Seidel_link_volume_and_cost(g_link_vector.size(), iteration_number, at, o, d, tau, ai);

										VehicleScheduleNetworks vsn;
										vsn.m_agent_type_no = at;
										vsn.m_time_interval_size = max(max(route_trip_time * 1.5, route_trip_time + 10), max(assignment.g_DemandPeriodVector[tau].get_time_horizon_in_min() * 1.5, assignment.g_DemandPeriodVector[tau].get_time_horizon_in_min() + 10));
										vsn.AllocateVSNMemory(assignment.g_number_of_nodes);
										vsn.BuildNetwork(assignment, tau, iteration_number);

										p_column->discrete_agent_path_vector[ai].path_link_sequence = vsn.g_optimal_time_dependenet_dynamic_programming(agent_path.o_node_no, agent_path.d_node_no, 0, tau, 10);

									}
								}
							}
						}
			}

		/*	if (assignment.g_pFileDebugLog != NULL)
							fprintf(assignment.g_pFileDebugLog, "CPU Running Time for SP  = %ld milliseconds\n", total_t);*/

							////step 3.2: calculate TD link travel time using TD inflow flow and capacity  
							//					start_t_1 = clock();


	}


	// step 4: column updating stage: for given column pool, update volume assigned for each column
	cout << "column pool updating with " << column_updating_iterations << " iterations" << endl;
	start_t = clock();
	g_column_pool_optimization(assignment, column_updating_iterations);

	// post route assignment aggregation
	if (assignment.assignment_mode != 0)
	{
		g_reset_and_update_link_volume_based_on_columns(g_link_vector.size(), iteration_number, false);  // we can have a recursive formulat to reupdate the current link volume by a factor of k/(k+1), and use the newly generated path flow to add the additional 1/(k+1)
	}
	else
	{
		g_reset_link_volume_without_columns(g_link_vector.size(), iteration_number, false);
	}

	update_link_travel_time_and_cost();  // initialization at the first iteration of shortest path

	end_t = clock();
	total_t = (end_t - start_t);
	cout << "Done!" << endl;

	cout << "CPU Running Time for column pool updating: " << total_t / 1000.0 << " s" << endl;

	start_t = clock();
	//step 5: output simulation results of the new demand 
	g_output_simulation_result(assignment);
	end_t = clock();
	total_t = (end_t - start_t);
	cout << "Output for assignment with " << assignment.g_number_of_K_paths << " iterations. Traffic assignment completes!" << endl;
	cout << "CPU Running Time for outputting simulation results: " << total_t / 1000.0 << " s" << endl;

	cout << "free memory.." << endl;
	cout << "done." << endl;

	g_node_vector.clear();

	for (int l = 0; l < g_link_vector.size(); l++)
	{
		g_link_vector[l].free_memory();
	}
	g_link_vector.clear();

	if (assignment.g_pFileDebugLog != NULL)
		fclose(assignment.g_pFileDebugLog);
	getchar();
	return 1;

}


