from time import time

## vertex_cover_tree initialises and calls the search tree
def vertex_cover_tree(graph):
    n = len(graph)
    cover = [None]*n
    return recursive_vertex_cover(graph, cover)

def recursive_vertex_cover(graph, cover):

    ############
    # ToDo: Program this part of the function

    #Check if it is possible to build a valid cover.

    # If not possible, return [1]*len(cover).

    #In another case, find two connected nodes u and v that are not in the cover.

    # If there are none, complete the cover by deciding whether the missing ones should be part of the cover.

    # of the cover or not and once this is done, return the complete cover.

    # otherwise continue with u and v
    
    # End of your code

    #The following opens the three branches of the search tree.

    # No change anything.

    ##############
        copy_cover = list(cover)
        cover[u] = 1
        cover[v] = 0
        c10 = recursive_vertex_cover(graph, cover)
        cover = list(copy_cover)
        cover[u] = 0
        cover[v] = 1
        c01 = recursive_vertex_cover(graph, cover)
        cover = list(copy_cover)
        cover[u] = 1
        cover[v] = 1
        c11 = recursive_vertex_cover(graph, cover)
        if c10.count(1) <= min(c01.count(1), c11.count(1)):
            return c10
        elif c01.count(1) <= c11.count(1):
            return c01
        else:
            return c11
    
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