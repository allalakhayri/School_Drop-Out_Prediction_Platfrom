import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
@Injectable({
  providedIn: 'root'
})
export class FileUploadService {

  apiUrl: string ="http://127.0.0.1:5000/upload";
  httpClient: any;
  constructor(private http: HttpClient) {}

uploadFile(formData: FormData) {
  const headers = new HttpHeaders();
  headers.append('Content-Type', 'multipart/form-data');
  headers.append('Accept', 'application/json');

  return this.http.post<any>(this.apiUrl, formData, { headers: headers });
}
}