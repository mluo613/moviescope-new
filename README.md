# iccv_moviescope
Extending the MovieScope project for Genre classification of Movie Trailers
Introducing a new dataset, stemming from IMDb5000 and MovieSummaries Dataset by CMU, and taking the intersection of the
two dataset calling it IMDb5k++.
<br>
IMDb5k++ contains Movie ID (assigned by us), Movie Title, Movie Trailer and Movie Plot Summary (from wikipedia).

<br>
Procedure to run: [wiki_imdb]
<br>
Skipping data gathering steps.
<br>
Dataset can be found as wiki_im_* [3878 trailers+plots] <br>
Run
    <ul>
        <li>prepare_data_wiki_im.py</li>
        <li>create_model_wiki_im.py</li>
        <li>test_model_multilabel_wiki_im.py</li>
    </ul>
<br>
<hr>
Procedure to run: [Old Version with 5027 trailers] == Deprecated

<br>
<ol>
    <li> Data fetching 
    <ul>
        <li>get_trailer_url</li>
        <li>download_trailer_script</li>
        <li>scrape_desc</li>
    </ul>
    </li>
    <li> Extracting VGG features
    <ul>
        <li>gather_features</li>
    </ul>
    </li>
    <li> Create deep model
    <ul>
        <li>prepare_data</li>
        <li>create_model</li>
    </ul>
    </li>
</ol>
