import pstats
#from pstats import SortKey
#p = pstats.Stats('profile_cartoon100k_cpu')
p = pstats.Stats('profile_202')
#p.strip_dirs().sort_stats(-1).print_stats()
#p.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)
p.strip_dirs().sort_stats('cumulative').print_stats(100)
