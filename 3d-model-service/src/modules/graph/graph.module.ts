import { Module } from '@nestjs/common';
import { GraphService } from './graph.service';
import { QueryBuilder } from './query.builder';
import { TransformerService } from './transformer.service';
import { Neo4jService } from './neo4j.service';

@Module({
  providers: [
    Neo4jService,
    GraphService,
    QueryBuilder,
    TransformerService
  ],
  exports: [
    GraphService,
    Neo4jService
  ]
})
export class GraphModule {}