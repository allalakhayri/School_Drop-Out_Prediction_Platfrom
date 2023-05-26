import { TestBed } from '@angular/core/testing';

import { PreprocessService } from './preprocess.service';

describe('PreprocessService', () => {
  let service: PreprocessService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(PreprocessService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
