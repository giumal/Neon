_carbon_benchmark_add(carbon_benchmark_halo_dist_faceonly halo_dist_faceonly.cu)
target_link_libraries(carbon_benchmark_halo_dist_faceonly PRIVATE carbon::carbon
                                                                  carbon_benchmark_common)
