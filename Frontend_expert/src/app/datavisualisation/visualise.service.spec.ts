import { TestBed } from '@angular/core/testing';

import { VisualiseService } from './visualise.service';

describe('VisualiseService', () => {
  let service: VisualiseService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(VisualiseService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
