class Metrics:
    """
    Class holding metrics options for the UI.
    """

    Spot = {
        "Time point": "Position T",
        "X coordinate": "Position X",
        "Y coordinate": "Position Y",
        "Condition": "Condition",
        "Replicate": "Replicate",
        "DISTANCE": "Distance",
        "Track length": "Track Length",
        "Net distance": "Net Distance",
        "Confinement ratio": "Confinement Ratio",
    }

    Track = {
        "Track ID": "Track ID",
        "Condition": "Condition",
        "Replicate": "Replicate",
        "Track length": "Track length", 
        "Net distance": "Net distance", 
        "Confinement ratio": "Confinement ratio",
        "Track points": "Number of points in track",
        "Speed mean": "Mean speed",
        "Speed median": "Median speed",
        "Speed max": "Max speed",
        "Speed min": "Min speed",
        "Speed std": "Speed standard deviation",
        "Direction mean (deg)": "Mean direction (degrees)",
        "Direction mean (rad)": "Mean direction (radians)",
        "Direction std (deg)": "Standard deviation (degrees)",
        "Direction std (rad)": "Standard deviation (radians)",
    }

    SpotAndTrack = Spot | Track

    Time = {
        "Track length mean": "Mean track length",
        "Net distance mean": "Mean net distance",
        "Confinement ratio mean": "Mean confinement ratio",
        "Track length median": "Median track length",
        "Net distance median": "Median net distance",
        "Confinement ratio median": "Median confinement ratio",
        "Distance min": "Minimum distance",
        "Direction std (deg)": "Standard deviation (degrees)",
        "Direction median (deg)": "Median direction (degrees)",
        "Direction mean (rad)": "Mean direction (radians)",
        "Direction std (rad)": "Standard deviation (radians)",
        "Direction median (rad)": "Median direction (radians)",
        "Speed min": "Minimum speed",
        "Speed max": "Maximum speed",
        "Speed mean": "Mean speed",
        "Speed std": "Speed standard deviation",
        "Speed median": "Median speed",
    }

    Lut = {
        "Track length": "Track length", 
        "Net distance": "Net distance", 
        "Confinement ratio": "Confinement ratio",
        "Track points": "Number of points in track",
        "Speed mean": "Mean speed",
        "Speed median": "Median speed",
        "Speed max": "Max speed",
        "Speed min": "Min speed",
        "Direction mean (deg)": "Mean direction (degrees)",
        "Direction mean (rad)": "Mean direction (radians)",
        }

    



class Styles:
    """
    Class holding color options for the UI.
    """

    ColorMode = [
        "random colors",
        "random greys",
        "differentiate conditions/replicates",
        "only-one-color",
        "greyscale LUT", 
        "jet LUT", 
        "brg LUT", 
        "hot LUT", 
        "gnuplot LUT", 
        "viridis LUT", 
        "rainbow LUT", 
        "turbo LUT", 
        "nipy-spectral LUT", 
        "gist-ncar LUT",
    ]

    PaletteQuantitative = [
        "greyscale", 
        "jet", 
        "brg", 
        "hot", 
        "gnuplot", 
        "viridis", 
        "rainbow", 
        "turbo", 
        "nipy-spectral", 
        "gist-ncar",
    ]

    PaletteQualitative = [
        "Set1",
        "Set2",
        "Set3",
        "tab10",
        "Accent",
        "Dark2",
        "Paired",
        "Pastel1",
        "Pastel2",
    ]

    PaletteQualitativeSeaborn = [
        "deep", 
        "muted", 
        "bright", 
        "pastel", 
        "dark", 
        "colorblind", 
        "Set1", 
        "Set2", 
        "Set3", 
        "tab10", 
        "tab20", 
        "tab20c",
    ]

    Color = [
        "red",
        "darkred",
        "firebrick",
        "crimson",
        "indianred",
        "salmon",
        "lightsalmon",
        "tomato",
        "coral",
        "orange",
        "darkorange",
        "gold",
        "khaki",
        "lemonchiffon",
        "peachpuff",
        "wheat",
        "tan",
        "peru",
        "chocolate",
        "sienna",
        "brown",
        "maroon",
        "yellow",
        "yellowgreen",
        "lawngreen",
        "greenyellow",
        "springgreen",
        "lightgreen",
        "palegreen",
        "mediumspringgreen",
        "mediumseagreen",
        "seagreen",
        "forestgreen",
        "green",
        "darkgreen",
        "olive",
        "teal",
        "turquoise",
        "mediumturquoise",
        "paleturquoise",
        "cyan",
        "darkcyan",
        "aqua",
        "deepskyblue",
        "skyblue",
        "lightblue",
        "powderblue",
        "steelblue",
        "dodgerblue",
        "blue",
        "mediumblue",
        "darkblue",
        "navy",
        "royalblue",
        "slateblue",
        "mediumslateblue",
        "slategray",
        "slategrey",
        "mediumorchid",
        "mediumpurple",
        "purple",
        "indigo",
        "darkviolet",
        "violet",
        "orchid",
        "plum",
        "magenta",
        "pink",
        "lightcoral",
        "rosybrown",
        "mistyrose",
        "lavender",
        "linen",
        "white",
        "snow",
        "whitesmoke",
        "lightgray",
        "lightgrey",
        "silver",
        "gray",
        "grey",
        "black",
    ]

    Background = [
        "light",
        "dark",
        "transparent",
    ]

    LineStyle = [
        "solid",
        "dashed",
        "dotted",
        "dashdot",
    ]



class Markers:
    """
    Class holding marker options for the UI.
    """

    PlotlyOpen = [
        "circle-open", 
        "square-open", 
        "triangle-open", 
        "star-open", 
        "diamond-open", 
        "pentagon-open",
    ]

    Emoji = [
        "cell",
        "scaled",
        "trains",
        "random",
        "farm",
        "safari",
        "insects",
        "birds",
        "forest",
        "aquarium",
    ]





class Modes:
    """
    Class holding modes for various functions for the UI.
    """

    Thresholding = [
        "Literal",
        "Normalized 0-1",
        "Quantile",
        "Relative to...",
        "Logarithmic",
    ]

    FitModel = {
        "Linear":         "(lambda x, a, b: a * x + b, [1, 0])",
        "Quadratic":       "(lambda x, a, b, c: a * x**2 + b * x + c, [1, 1, 0])",
        "Cubic":           "(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d, [1, 1, 1, 0])",
        "Logarithmic":     "(lambda x, a, b: a * np.log(x + 1e-9) + b, [1, 0])",
        "Exponential":     "(lambda x, a, b, c: a * np.exp(b * x) + c, [1, 0.1, 0])",
        "Logistic Growth": "(lambda x, L, k, x0: L / (1 + np.exp(-k * (x - x0))), [max(y), 1, np.median(x)]),",
        "Sine Wave":       "(lambda x, A, w, p, c: A * np.sin(w * x + p) + c, [np.std(y), 1, 0, np.mean(y)]),",
        "Gompertz":        "(lambda x, a, b, c: a * np.exp(-b * np.exp(-c * x)), [max(y), 1, 0.1]),",
        "Power Law":       "(lambda x, a, b: a * np.power(x + 1e-9, b), [1, 1])",
    }
    
    Interpolate = [
        "none",
        "basis",
        "basis-open",
        "basis-closed",
        "bundle",
        "cardinal",
        "cardinal-open",
        "cardinal-closed",
        "catmull-rom",
        "linear",
        "linear-closed",
        "monotone",
        "natural",
        "step",
        "step-before",
        "step-after",
    ]
    
    ExtentError = [
        "std", 
        "min-max",
    ]