import seaborn as sns
import matplotlib.pyplot as plt

class GroupPlot(object):

    def __init__(self, DataFrame, fig, x, y):
        self.df = DataFrame
        self.x  = x
        self.y  = y
        self.fig = fig
        self.set_defaults()

    def set_defaults(self):
        self.color_by  = None
        self.style_by  = None
        self.subplt_by = None
        self.subplt_cols = None
        self.fig_legend = None
        self.styles = None

    def get_numbers(self):
        if self.color_by:
            self.color_keys = self.df[self.color_by]
            self.Ncolors = len(self.color_keys.unique())
        if self.style_by:
            self.style_keys = self.df[self.style_by]
            self.Nstyles = len(self.style_keys.unique())
        if self.subplt_by:
            self.subplt_keys = self.df[self.subplt_by]
            self.Nsubplt = len(self.subplt_keys.unique())

    def get_color_dic(self):
        if self.color_by:
            if self.Ncolors > 6:
                #Default only has 6 colors, make new one if needed
                sns.set_palette(sns.color_palette("hls", self.Ncolors))
            color_dic = {}
            for i, key in enumerate(self.color_keys.unique()):
                color_dic[key] = sns.color_palette()[i]
            self.color_dic = color_dic

    def check_styles(self):
        if self.style_by:
            if not self.styles:
                self.styles = ['-', '--','-.', ':',]
            if  self.Nstyles > len(self.styles):
                raise ValueError('# of styles needed exceeds #'
                                + 'of styles available: currently '
                                + str(len(self.styles)) + ', needs '
                                + str(self.Nstyles) + '.')

    def get_style_dic(self):
        if self.style_by:
            style_dic = {}
            for i, key in enumerate(self.style_keys.unique()):
                style_dic[key] = self.styles[i]
            self.style_dic = style_dic

    def get_subplot_params(self):
    #Try to get close to a square layout by default
        if self.subplt_by:
            if not self.subplt_cols:
                subplt_cols = int(math.floor(np.sqrt(self.Nsubplt)))
            subplt_rows = self.Nsubplt / subplt_cols
            if self.Nsubplt % subplt_cols != 0:
                subplt_rows += 1
            self.subplt_rows = subplt_rows
            self.subplt_cols = subplt_cols

    def get_sub_dic(self):
        if self.subplt_by:
            sub_dic = {}
            for i, key in enumerate(self.subplt_keys.unique()):
                sub_dic[key] = [self.subplt_rows, self.subplt_cols, i+1]
            self.sub_dic = sub_dic

    def make_plots(self):
        self.lines = []
        self.labels = []

        for i in range(self.df.shape[0]):
            if self.subplt_by:
                subplot_val = self.sub_dic[self.subplt_keys[i]]
            else:
                subplot_val = [1,1,1]  # If not subplots, just 1 plot

            llabel = ''
            if self.style_by:
                lstyle = self.style_dic[self.style_keys[i]]
                llabel += self.style_keys[i]
                if self.color_by:
                    llabel += ', '
            else:
                lstyle = ''
            if self.color_by:    
                col = self.color_dic[self.color_keys[i]]
                llabel += self.color_keys[i]
            else:
                col = sns.color_palette()[0]


            self.labels.append(llabel)

            ax = self.fig.add_subplot(subplot_val[0],
                                 subplot_val[1],
                                 subplot_val[2])

            self.lines.append(ax.plot(self.df[self.x][i],
                                      self.df[self.y][i],
                                      lstyle, c=col, label=llabel))
            if self.subplt_by:
                plt.title(self.subplt_keys[i])
            plt.xlabel(self.x)
            plt.ylabel(self.y)

    def get_figlegend(self):
        unique_labels = []
        unique_lines  = []
        for i, label in enumerate(self.labels):
            if label not in unique_labels and label != '':
                unique_labels.append(label)
                unique_lines.append(self.lines[i][0])

        plt.figlegend((unique_lines), unique_labels,
                      bbox_to_anchor=(1.01, 0.5), loc='center left')

    def plot(self):
        self.get_numbers()
        self.get_subplot_params()
        self.check_styles()
        self.get_style_dic()
        self.get_color_dic()
        self.get_sub_dic()
        self.make_plots()
        if self.fig_legend:
            self.get_self.figlegend()
