import { Component, ViewChild, OnInit } from '@angular/core';
import { DrawableDirective } from './drawable.directive';

import * as tf from '@tensorflow/tfjs';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {

  linearModel: tf.Sequential;
  prediction: any;

  model: tf.Model;
  predictions: any;

  @ViewChild(DrawableDirective) canvas;

  ngOnInit() {
    
    this.loadModel();
  }


  

 


  //// LOAD PRETRAINED KERAS MODEL ////

  async loadModel() {
    this.model = await tf.loadModel('/assets/model.json');
  }

  async predict(imageData: ImageData) {

    const pred = await tf.tidy(() => {

      // Convert the canvas pixels to 
      let img = tf.fromPixels(imageData, 1);
      img = img.reshape([1,1, 28, 28]);
      img = tf.cast(img, 'float32');
	  const offset = tf.scalar(255.0);
      const normalized = tf.scalar(1.0).sub(img.div(offset));

      // Make and format the predications
      const output = this.model.predict(normalized) as any;

      // Save predictions on the component
      this.predictions = Array.from(output.dataSync()); 
    });

  }

}

