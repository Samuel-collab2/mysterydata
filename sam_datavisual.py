import pandas as pd
import matplotlib.pyplot as plt
from os import mkdir

OUTPUT_DIR = "sam_datavisual_dir"

try:
    mkdir(OUTPUT_DIR)
except FileExistsError:
    pass

# load data
data = pd.read_csv("trainingset.csv")

# NOTE for self ... Column with Claim amount (our label)
print(data[data.columns[19]])

# NOTE for self; Y axis CANNOT be categorical when plotting a line, bar, or histogram graph

# plot scatter, line, bar, and histogram for all combinations (that don't have categorical y-axis) with x-axis
types_of_graphs = ['scatter', 'line', 'bar', 'hist']

# Plot all features against each other with graph type specified in argument on line 59 (plt.savefig(...))
# Takes avg ~ 15s to finish running
for i in range(1, 20):
    for j in range((i + 1), 21):

        # if i == y, we will plot the feature against itself; so don't plot this
        if i == j:
            continue

        # jth index is out of scope
        if j == 20:
            continue

        x_axis = data.columns[i]
        y_axis = data.columns[j]

        data.plot(kind=types_of_graphs[0], x=x_axis, y=y_axis, color='red', title="{} vs {}".format(x_axis, y_axis),
                  legend=True)
        plt.savefig("{}/{}_{}vs{}.png".format(OUTPUT_DIR, types_of_graphs[0], x_axis, y_axis))
        print("j is now: ", j)
    print("i is now: ", i)


# Plot all features against each with ALL graph types
# !!! Currently doesn't work well; processing seems to slow down over time before all graphs can be completed !!!
# for z in range(0, 4):
#     for i in range(1, 20):
#         for j in range((i + 1), 21):
#             # if i == y, we will plot the feature against itself; so don't plot this
#             if i == j:
#                 continue
#
#             # jth index is out of scope
#             if j == 20:
#                 continue
#
#             x_axis = data.columns[i]
#             y_axis = data.columns[j]
#
#             data.plot(kind=types_of_graphs[z], x=x_axis, y=y_axis, color='red',
#                       title="{} vs {}".format(x_axis, y_axis), legend=True)
#             plt.savefig("{}/{}_{}vs{}.png".format(OUTPUT_DIR, types_of_graphs[z], x_axis, y_axis))
#             print("j is now: ", j)
#         print("i is now: ", i)



