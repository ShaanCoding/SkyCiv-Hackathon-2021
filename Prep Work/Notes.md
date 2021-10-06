# Drag n Drop Tutorial

- In this tutorial we won't be using dynamic data, this is just showing how we use the drag and drop api

```html
<!DOCTYPE html>
<html>
  <head>
    <title>Drag n Drop</title>
  </head>
  <body class="app">
    <div class="lists">
      <div class="list-item">List item 1</div>
      <div class="list-item">List item 2</div>
      <div class="list-item">List item 3</div>
    </div>
  </body>
</html>
```

```css
.lists {
  display: flex;
  flex: 1;
  width: 100%;
  overflow-x: scroll;
}

.lists .list {
  display: flex;
  flex-flow: column;
  flex: 1;

  width: 100%;
  min-width: 250px;
  max-width: 350px;
  height: 100%;
  min-height: 150px;

  background-color: rgba(0, 0, 0, 0.1);
  margin: 0 15px;
  padding: 8px;
  transition: all 0.2 linear;
}

.list .list .list-item {
}
```

```js
const list_items = document.querySelectorAll(".list-item");
const lists = document.querySelector(".list");

let draggedItem = null;

for (let i = 0; i < list_items.length; i++) {
  // Get current item in list

  const item = list_items[i];

  item.addEventListener("dragstart", (e) => {
    console.log("dragstart", e);
    draggedItem = this;
    // Want to hide it when dragging
    setTimeout(() => {
      item.style.display = "none";
    }, 0);
  });

  item.addEventListener("dragend", () => {
    console.log("dragend");
    // We don't want to do it too early, because we don't want it to drop and be null, we need a dropped event
    // We append to closest list
    setTimeout(() => {
      draggedItem.style.display = "block";
      draggedItem = null;
    }, 0);
  });

  for (let j = 0; j < list.length; j++) {
    const list = lists[j];

    // This doesn't work by normal, we need to overwrite some default actions on event listeners

    list.addEventListener("dragover", (e) => {
      e.preventDefault();
    });

    list.addEventListener("dragenter", (e) => {
      e.preventDefault();
      this.style.backroundColor = "rgba(0,0,0, 0.2)";
    });

    list.addEventListener('dragleave' (e) => {
      this.style.backroundColor = "rgba(0,0,0, 0.1)";
    })

    list.addEventListener("drop", (e) => {
      this.append(draggedItem);
      this.style.backroundColor = "rgba(0,0,0, 0.1)";
    });
  }
}
```

- We need to add drag and drop event listeners for each item

- We see it's invisible but we're still dragging it
