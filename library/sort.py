def zip_sort(lead_list, follow_list, comparator=lambda x: x[0], reverse=False):
    return zip(*sorted(zip(lead_list, follow_list), key=comparator, reverse=reverse))
