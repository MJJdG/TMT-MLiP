from matplotlib import pyplot as plt
import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont

HEIGHT = 236
WIDTH = 236

def get_n(df, class_map_df, field, n, top=True):
    top_graphemes = df.groupby([field]).size().reset_index(name='counts')['counts'].sort_values(ascending=not top)[:n]
    top_grapheme_roots = top_graphemes.index
    top_grapheme_counts = top_graphemes.values
    top_graphemes = class_map_df[class_map_df['component_type'] == field].reset_index().iloc[top_grapheme_roots]
    top_graphemes.drop(['component_type', 'label'], axis=1, inplace=True)
    top_graphemes.loc[:, 'count'] = top_grapheme_counts
    return top_graphemes

def image_from_char(char):
    image = Image.new('RGB', (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(image)
    myfont = ImageFont.truetype('bengaliai-cv19/kalpurush-fonts/kalpurush-2.ttf', 120)
    w, h = draw.textsize(char, font=myfont)
    draw.text(((WIDTH - w) / 2,(HEIGHT - h) / 3), char, font=myfont)

    return image

def print_top_10_roots(train_df, class_map_df):
    top_10_roots = get_n(train_df, class_map_df, 'grapheme_root', 10)
    print(top_10_roots)

#Doesn't work yet
def draw_top_10_roots(train_df, class_map_df):
    top_10_roots = get_n(train_df, class_map_df, 'grapheme_root', 10)

    f, ax = plt.subplots(2, 5, figsize=(16, 8))
    ax = ax.flatten()

    for i in range(10):
        ax[i].imshow(image_from_char(top_10_roots['component'].iloc[i]), cmap='Greys')