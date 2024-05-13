def choose_projects(whittle_indices):
    M = 3
    final_whittle = []
    for array in whittle_indices:
        print(array)
        finite_values = array[np.isfinite(array)]
        finite_sum = np.sum(finite_values)
        final_whittle.append(finite_sum)
    
    sorted_projects = np.argsort(final_whittle)
    active_projects = np.argsort(sorted_projects)[-M:]
    new_action = []
    for i in sorted_projects:
        if(i in active_projects):
            new_action.append(1)
        else:
            new_action.append(0)
    print(new_action)
    return final_whittle, sorted_projects, active_projects, new_action
