from time import time


# cover is a list of 0s, 1s and Nones.
# If i is the position of a 1, the i-th node is in the cover.
# If i is the position of a 0, the i-th node is not in the cover.
# If i is the position of a None, we do not know if the i-th node will be in the cover.

# partial_validity_check checks, given a possible cover and a graph,
# whether the cover nodes can cover all the edges of the graph or not.

def partial_validity_check(cover, graph):
    # TODO: Program the code of the function
    

# vertex_cover_tree initialises and calls search tree

def vertex_cover_tree(graph):
    n = len(graph)
    cover = [None]*n
    return recursive_vertex_cover(graph, cover)


# recursive_vertex_cover recursively builds the search tree 
def recursive_vertex_cover(graph, cover):

    ############
    # TODO: Program this part of the function
    #
    # Check if it is possible to construct a valid cover.
    # If not possible, return [1]*len(cover).
    # Otherwise, 
    # if the cover is complete, it returns it.
    # If incomplete, it finds a node v that is not in the cover.
    

    # End of your code
    # The following opens the two branches of the search tree.
    # Do not modify anything.
    ##############
        copy_cover = list(cover)
        cover[v] = 0
        c0 = recursive_vertex_cover(graph, cover)
        cover = list(copy_cover)
        cover[v] = 1
        c1 = recursive_vertex_cover(graph, cover)
        return c0 if c0.count(1) < c1.count(1) else c1

def test():
    
    g1 =  [[0, 1],
           [1, 0]]
       
    g2 = [[0, 1, 1],
          [1, 0, 0],
          [1, 0, 0]]
        
    
    g3 = [[0, 1, 1, 1, 1],
          [1, 0, 0, 0, 1],
          [1, 0, 0, 1, 1],
          [1, 0, 1, 0, 1],
          [1, 1, 1, 1, 0]]
    
        
    g4 = [[0, 1, 1, 1],
          [1, 0, 1, 0],
          [1, 1, 0, 1],
          [1, 0, 1, 0]]
    
       
    g5 = [[0, 1, 1, 0, 0, 0],
          [1, 0, 0, 1, 1, 0],
          [1, 0, 0, 1, 1, 1],
          [0, 1, 1, 0, 0, 1],
          [0, 1, 1, 0, 0, 0],
          [0, 0, 1, 1, 0, 0]]
    
    g6 = [[0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
          [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
          [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
          [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
          [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
          [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
          [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
          [1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1],
          [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
          [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1],
          [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]]

    

##    Uncomment to test partial_validity_check function
#    assert not partial_validity_check([0,0], g1)
#    assert not partial_validity_check([0,0,1], g2)
#    assert partial_validity_check([1,None,None], g2) 
#    assert partial_validity_check([0,None,None], g2) 
#    assert partial_validity_check([1,0,0], g2) 
#    assert partial_validity_check([1,1,0], g2) 
#    assert partial_validity_check([0,1,None], g2)
#    assert not partial_validity_check([0,None,0], g2) 
#    assert not partial_validity_check([0, 1, 1, 0, 1, 0], g5)
#    assert partial_validity_check([0, 1, 1, 1, 0, 0], g5) 
#    assert partial_validity_check([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1], g6)
    
###################################################################  
    

#    Uncomment to test recursive_vertex_cover function
#    assert vertex_cover_tree(g1) in [[1,0],[0,1]]
#    assert vertex_cover_tree(g2)  == [1,0,0]
#    assert vertex_cover_tree(g3) in [[1, 0, 1, 0, 1], 
#                                     [1, 0, 0, 1, 1]]
#    assert vertex_cover_tree(g4)  == [1, 0, 1, 0]
#    assert vertex_cover_tree(g5)  in  [[0, 1, 1, 1, 0, 0], 
#                                       [0, 1, 1, 0, 0, 1]]
#    
#    
#    assert vertex_cover_tree(g6) in [[1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
#                                     [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
#                                     [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
#                                     [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
#                                     [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
#                                     [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1],
#                                     [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
#                                     [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
#                                     [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
#                                     [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1],
#                                     [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
#                                     [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
#                                     [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
#                                     [1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
#                                     [1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0],
#                                     [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
#                                     [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
#                                     [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
#                                     [1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1],
#                                     [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1],
#                                     [1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
#                                     [1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
#                                     [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
#                                     [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0],
#                                     [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
#                                     [1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1],
#                                     [1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
#                                     [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
#                                     [1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1],
#                                     [1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1],
#                                     [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1]]
    
start_time = time()
test()
elapsed_time = time() - start_time   
print("Elapsed time: %0.10f seconds." % elapsed_time)      