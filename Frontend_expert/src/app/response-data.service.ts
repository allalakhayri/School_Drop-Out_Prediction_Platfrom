import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class ResponseDataService {
  [x: string]: any;
  calculateStatistics() {
    throw new Error('Method not implemented.');
  }
  map(arg0: (datum: { [x: string]: any; }) => any) {
    throw new Error('Method not implemented.');
  }
  private responseData: any;

  constructor() { }

  setResponseData(data: any) {
    this.responseData = data;
  }

  getResponseData() {
    return this.responseData;
  }
}