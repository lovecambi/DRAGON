#ifndef NETINFO_H
#define NETINFO_H

#include <vector>
#include <iostream>

using namespace std;

// Struct to store general information of hidden layers, does not contain details such as nodes' weights
struct layerInfoStr {
    int inputDim;           // Input dimension to the layer
    int numNodes;           // Number of nodes in this hidden layer
    string trsFuncName;     // Name of transfer function
    string cmbFuncName;     // Name of combine function
    layerInfoStr (int _inputDim = 0, int _numNodes = 0, string _trsFuncName = "", string _cmbFuncName = "") \
    : inputDim(_inputDim), numNodes(_numNodes), trsFuncName(_trsFuncName), cmbFuncName(_cmbFuncName)  {}
    layerInfoStr& operator=(const layerInfoStr& other) {
        if (this != &other) {
            inputDim = other.inputDim;
            numNodes = other.numNodes;
            trsFuncName = other.trsFuncName;
            cmbFuncName = other.cmbFuncName;
        }
        return *this;
    }
};

// Use to describe structure of a neural network at super node (hidden layer) scale
class netInfo {
public:
    int numSource;                      // Number of input sources to the network
    vector<int> srcDim;                 // Dimension of input sources to the network
    vector<string> srcDist;             // For fixed input is "fixed", if the input is random number, give its distribution name
    int numLayers;                      // Number of hidden layers in the network, does not include input layers
    vector<layerInfoStr> layerInfo;     // General information of hidden layers, does not contain details such as nodes' weights
    vector<vector<int>> schedulerFwd;   // Forward scheduler: divide layers into levels to apply parallelism for feed forawrd
    vector<vector<int>> schedulerBk;    // Backward scheduler: divide layers into levels to apply parallelism for back propagation
    vector<vector<int>> srcList;        // List of input source connected to the given node
    vector<vector<int>> parentList;     // List of node's parents given the node index
    vector<vector<int>> childList;      // List of node's children given the node index
public:
    // Constructor
    netInfo(int _numSource = 0, int _numLayers = 0) : numSource(_numSource), numLayers(_numLayers)
    {
        srcDim = vector<int>(numSource,0);
        srcDist = vector<string>(numSource,"");
        layerInfo = vector<layerInfoStr>(numLayers,layerInfoStr(0));
        srcList = vector<vector<int>>(numLayers, vector<int>(2,-1));
        parentList = vector<vector<int>>(numLayers,vector<int>());
        childList = vector<vector<int>>(numLayers,vector<int>());
    }
    
    // Constructor
    netInfo(vector<int> _srcDim, vector<string> _srcDist, vector<int> inputDims, vector<int> numNodes, vector<string> trsFuncName, vector<string> cmbFuncName, vector<vector<int>> src_layer_edges = vector<vector<int>>(), vector<vector<int>> layer_layer_edges = vector<vector<int>>()) : srcDim(_srcDim), srcDist(_srcDist)
    {
        numSource = (int) srcDim.size();
        assert( inputDims.size() == numNodes.size() && numNodes.size() == trsFuncName.size() && numNodes.size() == cmbFuncName.size() );
        numLayers = (int) numNodes.size();
        for (int i = 0; i < numLayers; i++) {
            layerInfoStr temp(inputDims[i], numNodes[i], trsFuncName[i], cmbFuncName[i]);
            layerInfo.push_back(temp);
        }
        srcList = vector<vector<int>>(numLayers, vector<int>(2,-1));
        parentList = vector<vector<int>>(numLayers,vector<int>());
        childList = vector<vector<int>>(numLayers,vector<int>());
        
        addEdge(src_layer_edges, layer_layer_edges);
        computeScheduler();
    }
    
    // Destructor
    ~netInfo()
    {
        clear();
    }
    
    // Clear memory for netInfo class
    void clear()
    {
        vector<int>().swap(srcDim);
        vector<string>().swap(srcDist);
        vector<layerInfoStr>().swap(layerInfo);
        vector<vector<int>>().swap(srcList);
        vector<vector<int>>().swap(parentList);
        vector<vector<int>>().swap(childList);
    }
    
    // = overload
    netInfo& operator=(const netInfo& other) {
        if (this != &other) {
            numSource = other.numSource;
            srcDim = other.srcDim;
            srcDist = other.srcDist;
            numLayers = other.numLayers;
            layerInfo = other.layerInfo;
            schedulerFwd = other.schedulerFwd;
            schedulerBk = other.schedulerBk;
            srcList = other.srcList;
            parentList = other.parentList;
            childList = other.childList;
        }
        return *this;
    }
    
    // Add a bunch of connections between input source and layer, or two layers
    void addEdge(vector<vector<int>> src_layer_edges, vector<vector<int>> layer_layer_edges)
    {
        for (vector<vector<int>>::iterator it = src_layer_edges.begin(); it != src_layer_edges.end(); it++)
            addEdge(*it, true);
        for (vector<vector<int>>::iterator it = layer_layer_edges.begin(); it != layer_layer_edges.end(); it++)
            addEdge(*it, false);
    }
    
    // After finishing adding edges, compute the scheduler
    void computeScheduler()
    {
        vector<int> doneList, temp;
        map<int, vector<int>> left;
        
        // Compute the forward scheduler
        // Find all layers that have no parent and only connected to input
        for (int i = 0; i < srcList.size(); i++)
            if (srcList[i][0] != -1 && parentList[i].empty()) {
                temp.push_back(i);
                doneList.push_back(i);
            }
            // If the layer has parents, store (layer, layer parents) as a remaining work
            else
                left[i] = parentList[i];
        // Push these layers as the first level of forward scheduler
        schedulerFwd.push_back(temp);

        while (!left.empty()) {
            temp.clear();
            // For each remaining work (layer, layer parents), if 'layer' is not done and all its parents are done
            // Remove this remaining work and add 'layer' to temp as a element of the new scheduler level
            for (map<int, vector<int>>::iterator iter = left.begin(); iter != left.end(); ) {
                if ( isElement(iter->first, doneList) && isSubset(iter->second, doneList) ) {
                    temp.push_back(iter->first);
                    iter = left.erase(iter);
                }
                else
                    iter++;
            }
            // Mark all added layers in current scheduler level as done
            for (int j = 0; j < temp.size(); j++)
                doneList.push_back(temp[j]);
            // Push these layers as new level of forward scheduler
            schedulerFwd.push_back(temp);
        }
        
        doneList.clear();
        temp.clear();
        
        // Compute the backword scheduler
        // Find all layers that have no child
        for (int i = 0; i < childList.size(); i++)
            if (childList[i].empty()) {
                temp.push_back(i);
                doneList.push_back(i);
            }
            // If the layer has children, store (layer, layer children) as a remaining work
            else
                left[i] = childList[i];
        // Push these layers as the first level of backward scheduler
        schedulerBk.push_back(temp);
        
        while (!left.empty()) {
            temp.clear();
            // For each remaining work (layer, layer children), if 'layer' is not done and all its children are done
            // Remove this remaining work and add 'layer' to temp as a element of the new scheduler level
            for (map<int, vector<int>>::iterator iter = left.begin(); iter != left.end(); ) {
                if ( isElement(iter->first, doneList) && isSubset(iter->second, doneList) ) {
                    temp.push_back(iter->first);
                    iter = left.erase(iter);
                }
                else
                    iter++;
            }
            // Mark all added layers in current scheduler level as done
            for (int j = 0; j < temp.size(); j++)
                doneList.push_back(temp[j]);
            // Push these layers as new level of forward scheduler
            schedulerBk.push_back(temp);
        }
    }
    
    // Given the child layer index cl and its parent layer index l
    // Return l's index in cl's parentList
    int getParentListIndex(int cl, int l)
    {
        int pc;
        for (pc = 0; pc < parentList[cl].size(); pc++ )
            if (parentList[cl][pc] == l)
                break;
        assert(pc != parentList[cl].size());
        return pc;
    }
    
    // Print for degbugging
    void print()
    {
        cout << endl << "num source = " << numSource << ", num layer = " << numLayers << endl;
        cout << "Input source dim: ";
        for (int i = 0; i < numSource; i++)
            cout << srcDim[i] << " ";
        cout << endl << endl;
        cout << "Input source dist: ";
        for (int i = 0; i < numSource; i++)
            cout << srcDist[i] << " ";
        cout << endl << endl;
        cout << "Layer info: " << endl;
        for (int i = 0; i < numLayers; i++)
            cout << layerInfo[i].inputDim << ", " << layerInfo[i].numNodes << ", " << layerInfo[i].trsFuncName << ", " << layerInfo[i].cmbFuncName << endl;
        cout << endl << "inputList: " << endl;
        for (int i = 0; i < numLayers; i++)
            if (srcList[i][0] != -1)
            {
                cout << "(src " << srcList[i][0] << " : layer " << i << "), ";
                if (srcList[i][1] != -1)
                    cout << "input " << srcList[i][0] << "is the " << srcList[i][1] + 1 << "-th parameter of combine function";
                cout << endl;
            }
        cout << endl;
        cout << endl << "parentList: " << endl;
        for (int i = 0; i < numLayers; i++) {
            cout << "parents of layer " << i << ": ";
            for (vector<int>::iterator it = parentList[i].begin(); it != parentList[i].end(); it++)
                cout << *it << " ";
            cout << endl;
        }
        cout << endl << "childList: " << endl;
        for (int i = 0; i < numLayers; i++) {
            cout << "children of layer " << i << ": ";
            for (vector<int>::iterator it = childList[i].begin(); it != childList[i].end(); it++)
                cout << *it << " ";
            cout << endl;
        }
        cout << endl;
        
        cout << endl << "schedulerFwd" << endl;
        for (int i = 0; i < schedulerFwd.size(); i++) {
            for (int j = 0; j < schedulerFwd[i].size(); j++)
                cout << schedulerFwd[i][j] << " ";
            cout << endl;
        }
        
        cout << endl << "schedulerBk" << endl;
        for (int i = 0; i < schedulerBk.size(); i++) {
            for (int j = 0; j < schedulerBk[i].size(); j++)
                cout << schedulerBk[i][j] << " ";
            cout << endl;
        }
    }
    
    // << overload: print the vector for debug
    friend ostream& operator<<(ostream& out, netInfo& nI)
    {
        // Save numSource and numLayers
        out << nI.numSource << " " << nI.numLayers << endl;
        
        // Save source dimensions
        for (int i = 0; i < nI.numSource; i++)
            out << nI.srcDim[i] << " ";
        out << endl << endl;
        
        // Save source distributions
        for (int i = 0; i < nI.numSource; i++)
            out << nI.srcDist[i] << " ";
        out << endl << endl;

        // Save layer info
        for (int i = 0; i < nI.numLayers; i++)
            out << nI.layerInfo[i].inputDim << " " << nI.layerInfo[i].numNodes << " " << nI.layerInfo[i].trsFuncName << " " << nI.layerInfo[i].cmbFuncName << endl;
        out << endl;
        
        // Save connections between source and layer
        for (int i = 0; i < nI.numLayers; i++)
            out << nI.srcList[i][0] << " " << nI.srcList[i][1] << endl;
        out << endl;
        
        // Save parent list
        for (int i = 0; i < nI.numLayers; i++) {
            if (!nI.parentList[i].empty())
                for (vector<int>::iterator it = nI.parentList[i].begin(); it != nI.parentList[i].end(); it++)
                    out << *it << " ";
            out << -1 << endl;
        }
        out << endl;
        
        // Save child list
        for (int i = 0; i < nI.numLayers; i++) {
            if (!nI.childList[i].empty())
               for (vector<int>::iterator it = nI.childList[i].begin(); it != nI.childList[i].end(); it++)
                    out << *it << " ";
            out << -1 << endl;
        }
        out << endl;
        
        // Save forward scheduler
        for (int i = 0; i < nI.schedulerFwd.size(); i++) {
            for (int j = 0; j < nI.schedulerFwd[i].size(); j++)
                out << nI.schedulerFwd[i][j] << " ";
            out << -1 << endl;
        }
        out << endl;
        
        // Save backward scheduler
        for (int i = 0; i < nI.schedulerBk.size(); i++) {
            for (int j = 0; j < nI.schedulerBk[i].size(); j++)
                out << nI.schedulerBk[i][j] << " ";
            out << -1 << endl;
        }

        return out;
    }
    
    // >> overload: print the vector for debug
    friend istream& operator>>(istream& in, netInfo& nI)
    {
        // Load numSource and numLayers
        in >> nI.numSource >> nI.numLayers;
        
        // Load source dimensions
        nI.srcDim.resize(nI.numSource);
        for (int i = 0; i < nI.numSource; i++)
            in >> nI.srcDim[i];
        
        // Load source distributions
        nI.srcDist.resize(nI.numSource);
        for (int i = 0; i < nI.numSource; i++)
            in >> nI.srcDist[i];
        
        // Load layer info
        nI.layerInfo.resize(nI.numLayers);
        for (int i = 0; i < nI.numLayers; i++)
            in >> nI.layerInfo[i].inputDim >> nI.layerInfo[i].numNodes >> nI.layerInfo[i].trsFuncName  >> nI.layerInfo[i].cmbFuncName;
        
        // Load connections between source and layer
        nI.srcList.resize(nI.numLayers);
        for (int i = 0; i < nI.numLayers; i++) {
            nI.srcList[i].resize(2);
            in >> nI.srcList[i][0] >> nI.srcList[i][1];
        }
        
        // Load parent list
        nI.parentList.resize(nI.numLayers);
        for (int i = 0; i < nI.numLayers; i++) {
            int temp_num = -1;
            in >> temp_num;
            // The number of -1 indicates end of vector, since we do not know how many parents ahead
            while (temp_num != -1) {
                nI.parentList[i].push_back(temp_num);
                in >> temp_num;
            }
        }
        
        // Load child list
        nI.childList.resize(nI.numLayers);
        for (int i = 0; i < nI.numLayers; i++) {
            int temp_num = -1;
            in >> temp_num;
            // The number of -1 indicates end of vector, since we do not know how many children ahead
            while (temp_num != -1) {
                nI.childList[i].push_back(temp_num);
                in >> temp_num;
            }
        }
        
        // Load forward scheduler
        for (int done_num = 0; done_num < nI.numLayers; ) {
            int temp_num = -1;
            vector<int> temp_vec;
            in >> temp_num;
            // The number of -1 indicates end of vector, since the size of each level is unknown ahead
            while (temp_num != -1) {
                temp_vec.push_back(temp_num);
                done_num++;
                in >> temp_num;
            }
            nI.schedulerFwd.push_back(temp_vec);
        }
        
        // Load backward scheduler
        for (int done_num = 0; done_num < nI.numLayers; ) {
            int temp_num = -1;
            vector<int> temp_vec;
            in >> temp_num;
            // The number of -1 indicates end of vector, since the size of each level is unknown ahead
            while (temp_num != -1) {
                temp_vec.push_back(temp_num);
                done_num++;
                in >> temp_num;
            }
            nI.schedulerBk.push_back(temp_vec);
        }
        
        return in;
    }
    
private:
    // Add connection between input source and layer, or two layers
    void addEdge( vector<int> edge, bool b_src = false)
    {
        if (b_src) {
            if (checkValidity(edge, true)) {
                srcList[edge[1]][0] = edge[0];
                if (edge.size() == 3)
                    srcList[edge[1]][1] = edge[2] - 1;
            }
        }
        else {
            if (checkValidity(edge, false)) {
                childList[edge[0]].push_back(edge[1]);
                parentList[edge[1]].push_back(edge[0]);
            }
        }
    }
    
    // Check for validity of the newly added edge
    // Self loops and multiple edges are not allowed, a layer can connect to one input source at most
    // For input source and parent layers of same child layer, their dimension / node number should be same
    // The logic is based on add IL edges before LL edges
    bool checkValidity(vector<int> edge, bool b_src)
    {
        bool is_valid = true;
        
        // Case 1: Adding IL edge, since IL edges are added before LL edges
        // Only need to check the case that multiple input sources connect to one layer
        if (b_src)
        {
            // Check bound validity
            assert(edge[0] >= 0 && edge[1] >= 0);
            assert(edge[0] < numSource && edge[1] < numLayers);
            
            int srcIdx = edge[0];
            int layerIdx = edge[1];
            is_valid = (srcList[layerIdx][0] == -1);
            if (!is_valid)
                cout << "IL (" << srcIdx << "," << layerIdx << "): The layer is already connected to an input source" << endl;
        }
        // Case 2: Adding LL edge
        else
        {
            // Check bound validity
            assert(edge[0] >= 0 && edge[1] >= 0);
            assert(edge[0] < numLayers && edge[1] < numLayers);
            // Do not allow self loop
            assert(edge[0] != edge[1]);
            
            int fromIdx = edge[0];
            int toIdx = edge[1];
            
            // Case 2-1: The to-layer has not been connect to any other layers, so it is free
            if (parentList[toIdx].empty())
                return true;
            // Case 2-2: The to layer has a parent layer
            else
            {
                // Check for multiple edge
                vector<int>::iterator it = find(parentList[toIdx].begin(), parentList[toIdx].end(), fromIdx);
                is_valid = (it == parentList[toIdx].end());
                if (!is_valid)
                    cout << "LL (" << fromIdx << "," << toIdx << "): This edge already exists" << endl;
            }

//            // Case 2-1: The to-layer has not been connect to any input source and other layers, so it is free
//            if (parentList[toIdx].empty() && srcList[toIdx] == -1)
//                return true;
//            // Case 2-2: The to layer has a parent layer
//            else if (!parentList[toIdx].empty())
//            {
//                // Check for multiple edge
//                vector<int>::iterator it = find(parentList[toIdx].begin(), parentList[toIdx].end(), fromIdx);
//                is_valid = (it == parentList[toIdx].end());
//                if (!is_valid)
//                    cout << "LL (" << fromIdx << "," << toIdx << "): This edge already exists" << endl;
//                // If it is not multiple edge, the new parent layer should have same node number as the first parent
//                else
//                {
//                    is_valid = (layerInfo[fromIdx].numNodes == layerInfo[parentList[toIdx][0]].numNodes);
//                    if (!is_valid)
//                        cout << "LL (" << fromIdx << "," << toIdx << "): Newly connected parent layer does not match previous connected parent layer" << endl;
//                }
//            }
//            // Case 2-3: The to-layer has no parent layer but have been connect to a input source
//            // Then the new parent layer's node number should match the input source dimension
//            else
//            {
//                is_valid = (layerInfo[fromIdx].numNodes  == srcDim[srcList[toIdx]]);
//                if (!is_valid)
//                    cout << "LL (" << fromIdx << "," << toIdx << "): Newly connected parent layer does not match connected input source" << endl;
//            }
            
        }
        return is_valid;
    }
    
    // whether a is a subset of A
    bool isElement(int a, vector<int> A) {
        return find(A.begin(), A.end(), a) == A.end();
    }
    
    // whether A is a subset of B
    bool isSubset(vector<int> A, vector<int> B) {
        sort(A.begin(), A.end());
        sort(B.begin(), B.end());
        return includes(B.begin(), B.end(), A.begin(), A.end());
    }
};

#endif
