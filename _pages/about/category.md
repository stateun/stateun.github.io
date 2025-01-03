---
layout: categories
title: "Categories"
permalink: /Categories/
author_profile: true
---

<div class="categories-list">
  <h2>Explore by Category</h2>
  <ul>
    {% for category in site.categories %}
    <li>
      <a href="{{ site.baseurl }}/categories/{{ category[0] | slugify }}/">
        {{ category[0] }}
      </a> ({{ category[1].size }})
    </li>
    {% endfor %}
  </ul>
</div>