import React, { constructor } from "react";

import "semantic-ui-css/semantic.min.css";
import axios from "axios";
import { Button, Divider, Form, Message, Segment, Tab } from "semantic-ui-react";
import { render } from "react-dom";



function App() {
  constructor(props) {
    super(props);
    this.state = {
      file: null
    };
  }

  fileInputRef = React.createRef();

  onFormSubmit = e => {
    e.preventDefault(); // Stop form submit
    this.fileUpload(this.state.file).then(response => {
      console.log(response.data);
    });
  };

  fileChange = e => {
    this.setState({ file: e.target.files[0] }, () => {
      console.log("File chosen --->", this.state.file);
    });
  };

  // Import datasources/schemas Tab 1
  fileUpload = file => {
    const url = "/some/path/to/post";
    const formData = new FormData();
    formData.append("file", file);
    const config = {
      headers: {
        "Content-type": "multipart/form-data"
      }
    };
    return put(url, formData, config);
  };

  // Export Schedules Tab 2
  fileExport = file => {
    // handle save for export button function
  };

  render() {
    const { file } = this.state;
    const panes = [
      {
        menuItem: "Import ",
        render: () => (
          <Tab.Pane attached={false}>
            <Message>Some message about offline usage</Message>
            <Form onSubmit={this.onFormSubmit}>
              <Form.Field>
                <Button
                  content="Choose File"
                  labelPosition="left"
                  icon="file"
                  onClick={() => this.fileInputRef.current.click()}
                />
                <input
                  ref={this.fileInputRef}
                  type="file"
                  hidden
                  onChange={this.fileChange}
                />
              </Form.Field>
              <Button type="submit">Upload</Button>
            </Form>
          </Tab.Pane>
        )
      }
    ];

  return (
    <div className="App">
      <Segment style={{ padding: "5em 1em" }} vertical>
        <Divider horizontal>OFFLINE USAGE</Divider>
        <Tab menu={{ pointing: true }} panes={panes} />
      </Segment>
    </div>
  );
}

export default App;
