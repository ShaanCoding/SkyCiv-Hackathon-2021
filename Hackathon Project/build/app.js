jQuery(document).ready(function () {
  const app_id = "report_generator";

  const config = {
    id: app_id, // The ID must be unique amongst other apps.
    name: "SkyCiv Report Generator",
    width: "600px",
    height: "600px",
    icon_img:
      "https://platform.skyciv.com/storage/images/logos/light/square-1.png",
    icon_img_square:
      "https://platform.skyciv.com/storage/images/logos/light/square-1.png",
    draggable: true,
    content: `<html>
  <head>
    <link
      href="https://dev.skyciv.com/assets/res/semantic/semantic.min.css"
      rel="stylesheet"
      type="text/css"
    />
    <style>
      /*Give your classes a unique suffix to ensure they dont interfere with existing styles*/
      .main-coolapp {
        /* Flex column to make all children stack downward */
        display: flex;
        flex-direction: column;
        /*Margin auto to center the main container*/
        margin: auto;
        /* Stop container getting too wide */
        max-width: 400px;
      }
      .h1-coolapp {
        text-align: center;
        color: black;
      }
    </style>
  </head>

  <body>
    <main class="main-coolapp">
      <h1 class="h1-coolapp">Hello SkyCiv Apps</h1>

      <button
        class="ui button"
        onclick="SKYCIV_APPS.${app_id}.customFunction()"
      >
        Notify
      </button>

      
    </main>
  </body>
</html>
`,
    onInit: function () {
      // This is called when the page loads the app.
      console.log("App has been initalised");
    },
    onFirstOpen: function () {
      // This is called on the first open of the App in the current S3D session.
    },
  };

  // This assigns the app into SKYCIV_APPS[app_id]
  new SKYCIV_APPS.create(config);

  // Get a reference to our app
  const app = SKYCIV_APPS[app_id];

  // Add custom functions to the application
  app.customFunction = function () {
    
  };

  app.secondCustomFunction = function () {
    console.log("The SECOND custom function was invoked.");

    // Get the model data
    const model = S3D.structure.get({ api_format: true });

    // Check if there is a model to screenshot
    const node_count = Object.keys(model.nodes).length;
    if (node_count > 0) {
      console.log("There is model data. Take a screenshot!");

      // A function to handle the data produced from the screenshot
      function callback(screenshotData) {
        // Using jQuery:
        // Get the img element
        const imgElement = $("#screenshot");
        // Set its src property to the screenshot data
        imgElement.attr("src", screenshotData);
        // Show the img element
        imgElement.show();
      }

      // Take a screenshot
      S3D.graphics.screenshot(callback);
    } else {
      // Tell the user to open a model
      SKYCIV.utils.alert.sideNotify({
        title: "No Model ⛔️",
        body: "Try opening a model before taking a screenshot.",
        time: 5000,
        auto_hide: true,
        theme: "dark",
      });
    }
  };

  // Initialize the app!
  app.init();
});
