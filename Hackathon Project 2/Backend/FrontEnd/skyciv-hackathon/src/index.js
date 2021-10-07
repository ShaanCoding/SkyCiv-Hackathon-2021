import React, { Component } from "react";
import ReactDOM from "react-dom";
import {
  Button,
  Segment,
  Divider,
  Tab,
  Table,
  Message,
  Checkbox,
  Form,
  Icon,
  Input,
  Dropdown,
  Dimmer,
  Loader,
  Label,
  LabelDetail,
} from "semantic-ui-react";

import "./styles.css";
import axios, { put } from "axios";

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      file: null,
    };
  }

  fileInputRef = React.createRef();

  onFormSubmit = (e) => {
    e.preventDefault(); // Stop form submit
    this.fileUpload(this.state.file);
  };

  // Import datasources/schemas Tab 1
  fileUpload = async (file) => {
    const url = "http://127.0.0.1:5000/image";
    const res = await axios.get(url);

    setTimeout(() => {
      window.location.replace(res.data.response);
    }, 2000);
  };

  // Export Schedules Tab 2
  fileExport = (file) => {
    // handle save for export button function
  };

  render() {
    return (
      <Segment style={{ padding: "5em 1em" }} vertical>
        <Divider horizontal>SKYCIV BRIDGE SCANNER</Divider>
        <Segment style={{}}>
          <Message>
            Welcome To SkyCiv Bridge Scanner, Please Select A Photo.
          </Message>
          <Form onSubmit={this.onFormSubmit}>
            <Form.Field>
              <Button
                content="Choose File"
                labelPosition="left"
                icon="file"
                onClick={() => this.fileInputRef.current.click()}
              />
              <input ref={this.fileInputRef} type="file" hidden />
            </Form.Field>
            <Button type="submit">Upload</Button>
          </Form>
        </Segment>
      </Segment>
    );
  }
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
