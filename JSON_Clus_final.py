import json
from anytree import Node, RenderTree, findall, LevelOrderIter, PostOrderIter
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog

q = 0
ct = 0
root = []
tdict = {} # the dictionary keeps track of all attributes of a document
keys = {} # contains the parent of each attribute. This is used to retrive data at any particular level
c = 1
r = []
nc = -1 
fptr = {}
lvl = Node('level')
y = 0
root.append(Node("root"+str(ct),parent= lvl))
ct = ct + 1
threshold = 0.2
clusters = [[[]] for row in range(8)] #holds all the cluster details
cluster_avg_len = [[0] for row in range(8)]
cluster_count = [0 for row in range(8)]
# cluster_avg_len[0].append(0) #set the avg len of first cluster to be zero
similarity_count = [[[]] for row in range(8)]  #to store attribute_count and match_count
sim_matrix = [[[]] for row in range(8)]
MyTreeRoot = Node('Root')
OptimalTree = Node('Optimal_Root')
cluster_tree = Node('clusterRoot')
highest_fscore = 0
highest_tval = 0
sim_graph = []
thr_graph = []

t_range = 8
t_val = 0.2

def setTrees(MyTreeRoot,cluster_tree,t_range,t_val):
    for i in range(t_range):
        Node(t_val,parent=MyTreeRoot)
        Node(t_val,parent=cluster_tree)
        t_val = (t_val*10 + 1)/ 10

setTrees(MyTreeRoot,cluster_tree,t_range,t_val)

def sort_this(temp):
    i = 0
    j = 0
    while j < len(temp)-1:
        i = 0
        while i < len(temp)-j-1:
            if len(temp[i+1]) < len(temp[i]):
                t = temp[i]
                temp[i] = temp[i+1]
                temp[i+1] = t
            elif len(temp[i+1]) == len(temp[i]):
                x = 0
                while x < len(temp[i]):
                    if temp[i+1][x] > temp[i][x]:
                        break
                    if temp[i+1][x] < temp[i][x]:
                        t = temp[i]
                        temp[i] = temp[i+1]
                        temp[i+1] = t
                        break
                    x = x+1
            i = i+1
        j = j+1
    return temp

def updateMatrix(c_num, match_count, parent_cnum,tval):    
    for x in sim_matrix[tval]:
        x[len(x):] = [0] # adding default value
    
    zero_arr = []
    for i in x:
        zero_arr.append(0)

    # if the cluster is independent
    if parent_cls_node == cluster_tree: 
        #make the last row full of zeroes.
        sim_matrix[tval].append(zero_arr)
    else:
        # update the distance with the parent
        sim_matrix[tval].append(zero_arr)
        sim_matrix[tval][c_num-1][parent_cnum-1] = match_count 
        sim_matrix[tval][parent_cnum-1][c_num-1] = match_count

        # update the distance with the siblings and update the dist for all its children
        # update_children(parent_cls_node,c_num)

        for ch in cluster_tree.children[tval].children:
            if c_num not in ch.cls: # this ch is sub_cls_root
                for i in ch.cls:
                    sim_matrix[tval][c_num-1][i-1] = 0
                    sim_matrix[tval][i-1][c_num-1] = 0
        
        x = sub_cls_root
        # now node has ispath=1. (child of sub_cls_root)
        while(1):
            if len(findall(x,filter_= lambda node: node.ispath in ([1]))) == 0:
                break
            
            for c in x.children:
                node = c
                if c.ispath == 1:
                    break
            # we have the node with ispath = 1
            # now all other nodes in that level have to be updated
            node.ispath = 0 #set it back to 0 for the next doc.
            update_children(x,node.c_num)
            x = node
            
def update_children(prnt,c_num):
    for c in prnt.children:
        if c.c_num != c_num:
            dist = min(sim_matrix[tval][prnt.c_num-1][c.c_num-1],sim_matrix[tval][prnt.c_num-1][c_num-1])
            sim_matrix[tval][c.c_num-1][c_num-1] = dist
            sim_matrix[tval][c_num-1][c.c_num-1] = dist
            #update the value for all its children
            for node in PostOrderIter(c):
                sim_matrix[tval][c_num-1][node.c_num-1] = dist
                sim_matrix[tval][node.c_num-1][c_num-1] = dist

def getdata(da,keys,k):
    parents = []
    parents.append(k)
    p = keys[k]
    while p!=0:
        parents.append(p)
        p = keys[p]
    d = da
    for pnt in reversed(parents):
        d = d[pnt]
    return d

def getFiles():
    rootfile = Tk()
    rootfile.title('File explorer')
    rootfile.filename = filedialog.askopenfilename(initialdir="C:/", title="Select a JSON dataset file", filetypes=(("json files","*.json*"),("all files","*.*")) )
    rootfile.quit
    global rf
    rf = rootfile.filename

while(1):
    #getFiles
    #open file
    #loop
    #ends, ask if you want to add another dataset to the current one
    
    getFiles()

    with open(rf) as f:
        data = json.load(f) 
        zed = len(data)

    while(threshold < 1):
        # MyTreeRoot = Node('Root')
        # #holds the structure of the clusters in tree format. Required for finding inter-cluster similarity.
        # cluster_tree = Node('clusterRoot') 
        tval = int(threshold * 10 -2)

        while(q < zed):
            # print("===============", q ,"==================")
            # print(sim_matrix)
            t2 = []
            keys = {}
            tdict = {}
            i = 0
            tl = []
            currentNode = MyTreeRoot.children[tval] #attribute tree
            currClusNode = cluster_tree.children[tval] # cluster tree
            sub_cls_root = cluster_tree.children[tval] #holds the position of the 1st child of the cluster tree root 
            attribute_count = 0 # no of attributes
            match_count = 0 #no of common attributes. If zero create a new cluster
            parent_cluster = 1 #assuming it is 1 initially
            cluster_updated = 0 #if this is 1 then no cluster must be created in both the att tree and the cluster tree.
            crt_Ind_Clstr = 1
            ccn_changed = 0

            for k in data[q].keys():
                # =========================== MY CODE START =======================================
                
                # regardless of whether the attribute is repeated or not it must be in the tree
                # check if this is this ispath already exists. Check from the root

                childrenCounter = 0
                createNewNode = 1
                attribute_count = attribute_count + 1

                for Nodechildren in currentNode.children:
                    if Nodechildren.name == str(k):
                        # make this the current node
                        NewcurrentNode = currentNode.children[childrenCounter]
                        createNewNode = 0
                        
                        #this code is to set the sub_cls_root
                        if crt_Ind_Clstr:
                            nc_counter = 0
                            for nch in currClusNode.children:
                                if nch.c_num == NewcurrentNode.c_num:
                                    sub_cls_root = currClusNode.children[nc_counter]
                                    sub_cls_root.ispath = 1
                                    currClusNode = sub_cls_root # so that the currClusNode doesnt stay at the root.
                                    ccn_changed = 1
                                    break
                                nc_counter = nc_counter+1

                        # when currClusNode changes in the above code, we shouldn't change it again in below code
                        #===== this if cond is for cluster tree =====
                        
                        if ccn_changed == 0:
                            if parent_cluster != Nodechildren.c_num:
                                cluschildcntr = 0
                                for nch in currClusNode.children:
                                    if nch.c_num == Nodechildren.c_num:
                                        newCurrClusNode = currClusNode.children[cluschildcntr]
                                        newCurrClusNode.ispath = 1
                                        break
                                    cluschildcntr = cluschildcntr+1
                                currClusNode = newCurrClusNode
                        else:
                            ccn_changed = 0
                        # ===== end =====
                        crt_Ind_Clstr = 0  #don't create a new subroot of this cluster
                        parent_cluster = Nodechildren.c_num
                        match_count = match_count + 1
                        break
                    childrenCounter = childrenCounter + 1

                # in if condition set the cluster number only if it is not updated.
                if createNewNode:
                    if match_count == 0 or match_count < (threshold * cluster_avg_len[tval][parent_cluster-1]):
                        if cluster_updated == 0:
                            c_num = cluster_count[tval] + 1
                            cluster_count[tval] = cluster_count[tval] + 1
                            cluster_updated = 1
                            NewChild = Node(k,parent=currentNode,c_num = c_num)
                            
                            # ===== this is for cluster tree =====
                            if crt_Ind_Clstr:
                                # print(q," creating in-dependent cluster")
                                NewClusChild = Node("clus_node",c_num = c_num,parent=cluster_tree.children[tval],cls=[],height = 0,ispath=1) #this condition is for independent clusters
                                NewClusChild.cls.append(c_num)
                                sub_cls_root = NewClusChild # because susb_cls_root is root node initially
                                crt_Ind_Clstr = 0
                                parent_cls_node = cluster_tree
                                # update the similarity matrix
                                if len(sim_matrix[tval]) == 0:
                                    sim_matrix[tval] = [[0]]
                                else:
                                    for x in sim_matrix[tval]:
                                        x[len(x):] = [0] # adding default value
                                    zero_arr = []
                                    for i in x:
                                        zero_arr.append(0)
                                    sim_matrix[tval].append(zero_arr)
                            else:
                                # print(q," creating dependent cluster")
                                parent_cls_node = currClusNode
                                NewClusChild = Node("clus_node",c_num = c_num,parent=currClusNode, height = currClusNode.height+1, ispath=1)
                                sub_cls_root.cls.append(c_num)      # add this c_num to cls of its root
                                # print(RenderTree(currClusNode))
                                # print(currClusNode.c_num)
                                updateMatrix(c_num,match_count,currClusNode.c_num,tval) #if parent is not root
                            
                            currClusNode = NewClusChild
                            # ===== end =====

                            parent_cluster = c_num
                            cluster_avg_len[tval].append(0)
                            clusters[tval].append([c_num])  #set the cluster
                        else:
                            #do not create a node in cluster tree
                            NewChild = Node(k,parent=currentNode,c_num = parent_cluster)    
                    else:
                        NewChild = Node(k,parent=currentNode,c_num = parent_cluster)
                    currentNode = NewChild
                else:
                    currentNode = NewcurrentNode

                #  =========================== MY CODE END =======================================
                if k not in tdict.keys() or k not in t2:
                    if k not in tdict.keys():
                        if q != 0:
                            y = y + 1
                            cv = "root" + str(y)
                            root.append(Node(k,parent= lvl))
                        tdict[k] = []
                        tdict[k].append(str(c))
                        tl.append(str(c))
                        temp = tdict[k]
                        fptr[str(c)] = Node(k,parent = root[y])
                        t2.append(k)
                        keys[k] = 0
                        c = c+1
                    else:
                        t2.append(k)
                        tl.append(str(tdict[k]))
                else:
                    tl.append(str(tdict[k]))

            keys[nc]= -1
            nc = nc -1
            tl.append('-1')
            t2.append(-1)
            #  =========================== MY CODE START =======================================
            childrenCounter = 0
            createNewNode = 1
            for Nodechildren in currentNode.children:
                if Nodechildren.name == '-1':
                    # make this the current node
                    NewcurrentNode = currentNode.children[childrenCounter]
                    createNewNode = 0
                    break
                childrenCounter = childrenCounter + 1

            if createNewNode:
                NewChild = Node('-1',parent=currentNode, c_num = parent_cluster)
                currentNode = NewChild
            else:
                currentNode = NewcurrentNode
            #  =========================== MY CODE END =======================================
            for k in t2:
                c1 = 1
                # this 'if' condition puts a -1 after each level
                if k == -1:
                    if t2[i-1] != -1 and i !=0:
                        t2.append(-1)
                        #  =========================== MY CODE START =======================================
                        childrenCounter = 0
                        createNewNode = 1
                        for Nodechildren in currentNode.children:
                            if Nodechildren.name == '-1':
                                # make this the current node
                                NewcurrentNode = currentNode.children[childrenCounter]
                                createNewNode = 0
                                break
                            childrenCounter = childrenCounter + 1

                        if createNewNode:
                            NewChild = Node('-1',parent=currentNode,c_num = parent_cluster)
                            currentNode = NewChild
                        else:
                            currentNode = NewcurrentNode
                        #  =========================== MY CODE END =======================================
                else:
                    d = getdata(data[q],keys,k)
                    if(isinstance(d,dict)):
                        for m,n in d.items():
                            #  =========================== MY CODE START =======================================
                            #add this attribute to the tree
                            childrenCounter = 0
                            createNewNode = 1
                            for Nodechildren in currentNode.children:
                                if Nodechildren.name == str(m):
                                    # make this the current node
                                    NewcurrentNode = currentNode.children[childrenCounter]
                                    createNewNode = 0
                                    #===== this if cond is for cluster tree =====
                                    #this code is useless if a cluster has been created
                                    if parent_cluster != Nodechildren.c_num:
                                        cluschildcntr = 0
                                        for nc in currClusNode.children:
                                            if nc.c_num == Nodechildren.c_num:
                                                newCurrClusNode = currClusNode.children[cluschildcntr]
                                                break
                                            cluschildcntr = cluschildcntr+1
                                        currClusNode = newCurrClusNode
                                    # ===== end =====
                                    parent_cluster = Nodechildren.c_num
                                    match_count = match_count + 1   
                                    break
                                childrenCounter = childrenCounter + 1

                            if createNewNode:
                                if cluster_updated == 0 and match_count < (threshold * cluster_avg_len[tval][parent_cluster-1]):
                                    c_num = cluster_count[tval] + 1
                                    cluster_count[tval] = cluster_count[tval] + 1
                                    cluster_updated = 1
                                    
                                    #clus tree code. There wont be a case of crt_ind_cls = 1
                                    NewClusChild = Node("clus_node",c_num = c_num,parent=currClusNode, height = currClusNode.height+1,ispath = 1)
                                    sub_cls_root.cls.append(c_num)      # add this c_num to cls of its root
                                    parent_cls_node = currClusNode
                                    updateMatrix(c_num,match_count,currClusNode.c_num,tval)
                                    currClusNode = NewClusChild

                                    NewChild = Node(m,parent=currentNode,c_num = c_num)
                                    parent_cluster = c_num
                                    cluster_avg_len[tval].append(0)
                                    clusters[tval].append([c_num]) #this indicates the cluster number
                                else: 
                                    NewChild = Node(m,parent=currentNode, c_num = parent_cluster)
                                currentNode = NewChild
                            else:
                                currentNode = NewcurrentNode

                            #  =========================== MY CODE END =======================================
                            if m not in tdict.keys():
                                tdict[m] = []
                                for it in tdict.values():
                                    for im in it:
                                        if(str(tdict[k][-1]) + '.' + str(c1) == im):
                                            c1 = c1 + 1
                                tdict[m].append(str(tdict[k][-1]) + '.' + str(c1))
                                fptr[str(tdict[m][-1])] = Node(m,parent = fptr[tdict[k][-1]])
                                tl.append(tdict[m])
                                t2.append(m)
                                keys[m] = k
                                c1 = c1+1
                            else:
                                tl.append(tdict[m])
                                t2.append(m)
                                keys[m] = k
                            attribute_count = attribute_count+1
                    elif(isinstance(d,list)):
                        #if it's not an object but an array of objects
                        for s in d:
                            #since d is an array here
                            if(isinstance(s,dict)):
                                for m,n in s.items():
                                    #  =========================== MY CODE START =======================================
                                    #add this attribute to the tree
                                    childrenCounter = 0
                                    createNewNode = 1
                                    for Nodechildren in currentNode.children:
                                        if Nodechildren.name == str(m):
                                            # make this the current node
                                            NewcurrentNode = currentNode.children[childrenCounter]
                                            createNewNode = 0
                                            parent_cluster = Nodechildren.c_num
                                            match_count = match_count + 1 
                                            break
                                        childrenCounter = childrenCounter + 1

                                    if createNewNode:
                                        if cluster_updated == 0 and match_count < (threshold * cluster_avg_len[tval][parent_cluster-1]):
                                            c_num = cluster_count[tval] + 1
                                            cluster_count[tval] = cluster_count[tval] + 1
                                            cluster_updated = 1
                                            
                                            #clus tree code. There wont be a case of crt_ind_cls = 1
                                            NewClusChild = Node("clus_node",c_num = c_num,parent=currClusNode, height = currClusNode.height+1,ispath = 1)
                                            sub_cls_root.cls.append(c_num)      # add this c_num to cls of its root
                                            updateMatrix(c_num,match_count,currClusNode.c_num,tval)
                                            currClusNode = NewClusChild

                                            NewChild = Node(m,parent=currentNode,c_num = c_num)
                                            parent_cluster = c_num
                                            cluster_avg_len[tval].append(0)
                                            clusters[tval].append([c_num])
                                        else: 
                                            NewChild = Node(m,parent=currentNode, c_num = parent_cluster)
                                        currentNode = NewChild
                                    else:
                                        currentNode = NewcurrentNode
                                    #  =========================== MY CODE END =======================================

                                    if m not in tdict.keys():
                                        tdict[m] = []
                                        for it in tdict.values():
                                            for im in it:
                                                if(str(tdict[k][0]) + '.' + str(c1) == it):
                                                    c1 = c1 + 1
                                        tdict[m].append(str(tdict[k][0]) + '.' + str(c1))
                                        fptr[str(tdict[m][-1])] = Node(m,parent=fptr[tdict[k][-1]])
                                        tl.append(tdict[m])
                                        #append m only if it's value is an object or array
                                        if(isinstance(n,dict)):
                                            t2.append(m)
                                        elif(isinstance(n,list)):
                                            t2.append(m)
                                        keys[m] = k
                                        c1 = c1+1
                                    else:
                                        tl.append(tdict[m])
                                        if(isinstance(n,dict)):
                                            t2.append(m)
                                        elif(isinstance(n,list)):
                                            t2.append(m)
                                        keys[m] = k
                i = i+1
            
            # adding the document to the cluster and updating the cluster avg length
            if cluster_avg_len[tval][parent_cluster-1] == 0:
                cluster_avg_len[tval][parent_cluster-1] = attribute_count
                match_count = attribute_count #since this formed a new cluster
            else:
                cluster_avg_len[tval][parent_cluster-1] = (cluster_avg_len[tval][parent_cluster-1] + attribute_count)/2
                        
            #add the doc to its respective cluster
            clusters[tval][parent_cluster].append(q+1)
            similarity_count[tval].append([q+1])
            similarity_count[tval][q+1].append(attribute_count)
            similarity_count[tval][q+1].append(match_count)

            t2.pop()
            tl.append('-1')
            q = q + 1
            # print(RenderTree(cluster_tree))
        
        print("============== THRESHOLD: ",threshold," ==================")
        print(clusters[tval])
        print(cluster_avg_len)
        print(similarity_count[tval])
        #calculate the f1-score:
        first_cluster = 1
        tot_clus = len(clusters[tval]) - 1
        sim_pro = 1
        sim_sum = 0

        clstrPntr = 1
        simMatPntr = 0
        for i in range(tot_clus):
            # i starts from 0
            #========= for P in f1 score==============
            first_doc = 1
            for doc_no in clusters[tval][clstrPntr+i]:
                #ignore the first one (its cluster number)
                #total attribute and match counts
                tot_att = 0
                tot_mat = 0
                if first_doc:
                    first_doc = 0
                else:
                    #this gives the similarity of the cluster
                    tot_att = tot_att + similarity_count[tval][doc_no][1]
                    tot_mat = tot_mat + similarity_count[tval][doc_no][2]
            # the formula used to calculate similarity is: (attr = attributes)
            # P = (total no attr that match with the existing attr of that cluster) / ( tot no of attributes in that cluster )
            p = tot_mat / tot_att

            # ========= for R =========================
            totDiff = 0
            for d in sim_matrix[tval][simMatPntr+i]:
                totDiff = totDiff+d

            sim_pro = sim_pro * (totDiff+1) * (1/p)
            sim_sum = sim_sum + (totDiff+1) + (1/p)
        
        # calculating the metric which is similar to f1-score. This helps us in deciding whether the overall cluster similarity score is good enough 
        # to pick a particular threshold for the given sample.
        sim_score = (2 * sim_pro) / sim_sum
        print(sim_score)
        sim_graph.append(sim_score)
        thr_graph.append(threshold)
        if sim_score > highest_fscore:
            highest_fscore = sim_score
            highest_tval = tval
            #save the tree structure 
            OptimalTree = None
            OptimalTree = Node('Optimal')
            NewTree = MyTreeRoot
            NewTree.parent = OptimalTree
        
        threshold = (threshold*10 + 1)/ 10
        # clusters = [[] for row in range(1)]
        # similarity_count = [[] for row in range(1)]
        # sim_matrix =[[] for row in range(1)]
        # cluster_avg_len = [[]]
        # cluster_avg_len.append(0)
        # cluster_count = 0
        q = 0

    # the first child of optimal tree is root, This line prints the optimal tree.
    print(RenderTree(OptimalTree.children[0].children[highest_tval]))
    threshold = 0.2
    nextDS = input("Do you want to add another dataset to the existing one? Press 'y' or 'n'. ")
    if nextDS == "n" or nextDS == "N":
        break
    #plotting the graph
    # plt.plot(thr_graph,sim_graph)
    # plt.xlabel('Threshold')
    # plt.ylabel('F1-score')
    # plt.show()