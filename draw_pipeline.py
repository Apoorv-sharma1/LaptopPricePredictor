import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(16, 13), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Helper to draw rounded boxes
    def add_box(x, y, w, h, text, facecolor, edgecolor, fontcolor='black', fontsize=11, weight='bold', ls='-'):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5", 
                                     facecolor=facecolor, edgecolor=edgecolor, linewidth=2, linestyle=ls)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', color=fontcolor, 
                fontsize=fontsize, fontweight=weight, wrap=True)

    # Helper to draw arrows
    def add_arrow(x_start, y_start, x_end, y_end):
        ax.annotate("", xy=(x_end, y_end), xytext=(x_start, y_start),
                    arrowprops=dict(arrowstyle="->", color="black", lw=2))

    # --- Title ---
    ax.text(50, 96, "DATA ANALYSIS & PREPROCESSING PIPELINE", ha='center', va='center', fontsize=26, fontweight='bold', color='black')
    
    # Decorative lines
    ax.plot([5, 20], [96, 96], color='#111', lw=2)
    ax.plot([80, 95], [96, 96], color='#111', lw=2)
    ax.plot([20], [96], marker='o', markersize=6, color='#111')
    ax.plot([80], [96], marker='o', markersize=6, color='#111')

    # --- Start ---
    add_box(38, 88, 24, 4, "▶ Start Data Pipeline", '#FFF0F5', '#FF1493', '#FF1493', 15)
    add_arrow(50, 88, 50, 84)

    # ==========================================
    # --- EDA SECTION ---
    # ==========================================
    add_box(5, 63, 90, 21, "", '#F8FBFF', '#1E90FF', ls='-')
    ax.text(50, 82, "EXPLORATORY DATA ANALYSIS (EDA)", ha='center', va='center', fontsize=15, fontweight='bold', color='#1E90FF')

    # EDA Steps
    add_box(8, 66, 13, 11, "Data Acquisition\n(laptop_data.csv)", 'white', '#1E90FF', '#1E90FF', 10)
    add_arrow(21, 71.5, 25, 71.5)
    
    add_box(25, 66, 13, 11, "Inspect Data\nStructure", 'white', '#1E90FF', '#1E90FF', 10)
    add_arrow(38, 71.5, 42, 71.5)
    
    add_box(42, 66, 13, 11, "Descriptive\nStatistics", 'white', '#1E90FF', '#1E90FF', 10)
    add_arrow(55, 71.5, 59, 71.5)
    
    add_box(59, 66, 13, 11, "Analyze Feature\nDistributions", 'white', '#1E90FF', '#1E90FF', 10)
    add_arrow(72, 71.5, 76, 71.5)
    
    add_box(76, 66, 13, 11, "Correlation\nMatrix Analysis", 'white', '#1E90FF', '#1E90FF', 10)

    # Arrow down
    add_arrow(50, 63, 50, 58)

    # ==========================================
    # --- PREPROCESSING SECTION ---
    # ==========================================
    add_box(5, 14, 90, 44, "", '#FFFDF8', '#FF8C00')
    ax.text(50, 55, "DATA PREPROCESSING PIPELINE", ha='center', va='center', fontsize=15, fontweight='bold', color='#D2691E')

    # Step 1: Cleaning
    add_box(8, 41, 84, 10, "", '#F6FFF6', '#32CD32', ls='--')
    ax.text(50, 49, "Data Cleaning & Formatting", ha='center', va='center', fontsize=12, fontweight='bold', color='#228B22')
    add_box(14, 42.5, 20, 5, "Drop Unnamed: 0", 'white', '#32CD32', 'black', 10)
    add_arrow(34, 45, 39, 45)
    add_box(39, 42.5, 22, 5, "Clean RAM (remove GB)", 'white', '#32CD32', 'black', 10)
    add_arrow(61, 45, 66, 45)
    add_box(66, 42.5, 20, 5, "Clean Weight (remove kg)", 'white', '#32CD32', 'black', 10)

    # Step 2: Feature Engineering
    add_box(8, 28, 84, 11, "", '#F6FFF6', '#32CD32', ls='--')
    ax.text(50, 37, "Feature Engineering & Extraction", ha='center', va='center', fontsize=12, fontweight='bold', color='#228B22')
    add_box(10, 29.5, 17, 6, "Screen Features\n(PPI, Touch)", 'white', '#32CD32', 'black', 10)
    add_arrow(27, 32.5, 30, 32.5)
    add_box(30, 29.5, 15, 6, "CPU Brand\nGrouping", 'white', '#32CD32', 'black', 10)
    add_arrow(45, 32.5, 48, 32.5)
    add_box(48, 29.5, 19, 6, "Memory Split\n(HDD, SSD, Flash)", 'white', '#32CD32', 'black', 10)
    add_arrow(67, 32.5, 70, 32.5)
    add_box(70, 29.5, 18, 6, "GPU Brand & OS\nGrouping", 'white', '#32CD32', 'black', 10)

    # Step 3: Model Preparation
    add_box(8, 16, 84, 10, "", 'white', '#FF8C00', ls='-')
    add_box(12, 17.5, 16, 7, "Log Transform\nPrice Target", 'white', '#FF8C00', 'black', 10)
    add_arrow(28, 21, 32, 21)
    add_box(32, 17.5, 16, 7, "One-Hot\nEncoding", 'white', '#FF8C00', 'black', 10)
    add_arrow(48, 21, 52, 21)
    add_box(52, 17.5, 16, 7, "Data Splitting\n(85/15)", 'white', '#FF8C00', 'black', 10)
    add_arrow(68, 21, 72, 21)
    add_box(72, 17.5, 16, 7, "Standardization\n(StandardScaler)", 'white', '#FF8C00', 'black', 10)

    # Arrow to End
    add_arrow(50, 14, 50, 10)

    # --- End ---
    add_box(35, 4, 30, 6, "🎓 Ready for Model Training", '#FFF0F5', '#FF1493', '#FF1493', 15)

    plt.savefig('pipeline_flowchart.png', bbox_inches='tight')
    print("Flowchart successfully saved as 'pipeline_flowchart.png'!")

if __name__ == '__main__':
    draw_flowchart()
