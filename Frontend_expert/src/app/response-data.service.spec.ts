import { TestBed } from '@angular/core/testing';

import { ResponseDataService } from './response-data.service';

describe('ResponseDataService', () => {
  let service: ResponseDataService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ResponseDataService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
